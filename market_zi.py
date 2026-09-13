"""
Zero-intelligence / Poisson control simulator (Part 2), implementing
Smith, Farmer, Gillemot & Krishnamurthy 2003 ("Statistical theory of the
continuous double auction") directly and calibrating its five parameters
(alpha, mu, delta, dp, sigma) from the same real INTC data used to fit the
queue-reactive model, so it is a fair, matched-rate control: same order
size and roughly the same aggregate flow, but with NO dependence of any
event's rate on the current queue state -- the point of contrast with
market_qr.py.

Calibration:
  delta (per-order cancel hazard) = the queue-reactive model's fitted
    cancellation slope (a_C / Q0). This is an exact correspondence: Smith's
    lambda_C(q) = delta * q is the same functional form iex_calibrate.py
    fits, so delta IS that slope, not a re-estimation.
  sigma (chunk size) = 100 shares, the median real INTC trade size.
  mu (market order rate, shares/time) = a_MO (fitted event rate, /sec) * sigma.
  alpha (limit order rate, shares/price/time): calibrated via Smith's own
    closed-form mean-spread approximation s ~ mu / (2*alpha) (Section
    II.A), solved for alpha against the REAL median observed INTC spread,
    rather than converting our touch-concentrated queue-reactive arrival
    rate directly -- real order placement clusters much more tightly at
    the touch than Smith's uniform-over-all-prices assumption predicts, so
    a naive rate/tick-width conversion badly underestimates the spread
    (verified: it gives a sub-tick pc, i.e. a degenerate near-zero spread).
    Targeting the real spread directly is the more defensible calibration
    of the two, and is exactly the quantity Smith's own dimensional
    analysis says alpha controls.

The model assumes limit orders can land at any price beyond the opposite
touch out to infinity; we truncate to a finite tick domain (a few dozen
characteristic prices pc wide) since only prices within a few pc of the
touch matter for any statistic we compute (Smith Section II.B, explicit
caveat: "results are potentially relevant to real markets only when p is
at most a few times pc").
"""
import csv
import glob
import json
import os

import numpy as np


def calibrate_zi_params(qr_params, archive_dir='archive', symbol='INTC',
                         tick_size=0.005, sigma=100.0):
    delta = qr_params['a_C'] / qr_params['Q0']
    mu = qr_params['a_MO'] * sigma
    s_target = _median_observed_spread(archive_dir, symbol)
    alpha = mu / (2.0 * s_target)
    return {'alpha': alpha, 'mu': mu, 'delta': delta, 'dp': tick_size,
            'sigma': sigma, 's_target': s_target}


def _median_observed_spread(archive_dir, symbol, sample_every=200000):
    """Cheap proxy for the real median bid/ask spread: replay one day's
    price-level stream in order, tracking best bid/ask, sampling
    periodically (full tick-by-tick spread reconstruction isn't needed for
    a single calibration number)."""
    day_dirs = sorted(glob.glob(os.path.join(archive_dir, '*')))
    for d in day_dirs:
        prl_files = glob.glob(os.path.join(d, '*_prl.csv'))
        if not prl_files:
            continue
        spreads = []
        books = {'0': {}, '1': {}}
        n = 0
        with open(prl_files[0]) as f:
            for row in csv.DictReader(f):
                if row['Symbol'] != symbol:
                    continue
                side = row[' Buy_Ask Flag'].strip()
                price = float(row['Price'])
                size = float(row['Size'])
                book = books[side]
                if size <= 0:
                    book.pop(price, None)
                else:
                    book[price] = size
                n += 1
                if n % sample_every == 0 and books['0'] and books['1']:
                    bb, ba = max(books['0']), min(books['1'])
                    if ba > bb:
                        spreads.append(ba - bb)
        if spreads:
            spreads.sort()
            return spreads[len(spreads) // 2]
    raise RuntimeError(f'no archived day with {symbol} data found under {archive_dir}')


class ZeroIntelligenceBook:
    def __init__(self, alpha, mu, delta, dp, sigma, n_ticks=300, rng=None):
        self.alpha = alpha
        self.mu = mu
        self.delta = delta
        self.dp = dp
        self.sigma = sigma
        self.n_ticks = n_ticks
        self.rng = rng if rng is not None else np.random.default_rng()

        # bid_book[i] rests at mid0 - (i+1)*dp; ask_book[i] rests at mid0 + (i+1)*dp.
        # Fixed reference grid -- valid for short bursts around the calibration
        # point; long-running use would need periodic re-centering (Part 3).
        self.bid_book = np.zeros(n_ticks)
        self.ask_book = np.zeros(n_ticks)
        self.mid0 = None

        self.lo_chunk_rate = alpha / sigma  # chunks per unit price per unit time
        self.mo_chunk_rate = mu / sigma / 2.0  # per side

    def reset(self, mid_price=100.0, burn_in_events=4000):
        self.mid0 = mid_price
        pc = self.mu / (2 * self.alpha)
        steady_depth = self.alpha / self.delta  # asymptotic depth, shares/price (Table III)
        # seed a plausible starting profile so burn-in is short: roughly
        # steady_depth*dp shares per tick, decaying toward the touch as
        # Smith Fig. 3 shows for large epsilon.
        far = steady_depth * self.dp
        self.bid_book[:] = far
        self.ask_book[:] = far
        for _ in range(burn_in_events):
            self.step_background()
        return self.snapshot()

    def touch_index(self, side):
        book = self.bid_book if side == 'bid' else self.ask_book
        nz = np.nonzero(book)[0]
        return int(nz[0]) if len(nz) else None

    def step_background(self):
        bid_touch = self.touch_index('bid')
        ask_touch = self.touch_index('ask')
        n = self.n_ticks

        lo_bid_rate = self.lo_chunk_rate * self.dp * n
        lo_ask_rate = self.lo_chunk_rate * self.dp * n
        cancel_bid_total = self.delta * self.bid_book.sum() / self.sigma
        cancel_ask_total = self.delta * self.ask_book.sum() / self.sigma
        mo_buy_rate = self.mo_chunk_rate if ask_touch is not None else 0.0
        mo_sell_rate = self.mo_chunk_rate if bid_touch is not None else 0.0

        rates = np.array([lo_bid_rate, lo_ask_rate, cancel_bid_total,
                           cancel_ask_total, mo_buy_rate, mo_sell_rate])
        total_rate = rates.sum()
        if total_rate <= 0:
            return 0.0, {'type': 'noop'}

        dt = self.rng.exponential(1.0 / total_rate)
        choice = self.rng.choice(6, p=rates / total_rate)

        if choice == 0:
            i = self.rng.integers(0, n)
            self.bid_book[i] += self.sigma
            event = {'type': 'LO', 'side': 'bid', 'level': i}
        elif choice == 1:
            i = self.rng.integers(0, n)
            self.ask_book[i] += self.sigma
            event = {'type': 'LO', 'side': 'ask', 'level': i}
        elif choice == 2:
            i = self._weighted_level(self.bid_book)
            removed = min(self.sigma, self.bid_book[i])
            self.bid_book[i] -= removed
            event = {'type': 'cancel', 'side': 'bid', 'level': i, 'size': removed}
        elif choice == 3:
            i = self._weighted_level(self.ask_book)
            removed = min(self.sigma, self.ask_book[i])
            self.ask_book[i] -= removed
            event = {'type': 'cancel', 'side': 'ask', 'level': i, 'size': removed}
        elif choice == 4:
            fills = self._market_order(self.ask_book, self.sigma)
            event = {'type': 'MO', 'side': 'buy', 'size': self.sigma, 'fills': fills}
        else:
            fills = self._market_order(self.bid_book, self.sigma)
            event = {'type': 'MO', 'side': 'sell', 'size': self.sigma, 'fills': fills}

        return dt, event

    def _weighted_level(self, book):
        total = book.sum()
        if total <= 0:
            return 0
        p = book / total
        return int(self.rng.choice(len(book), p=p))

    def _market_order(self, book, size):
        fills = []
        remaining = size
        i = 0
        while remaining > 0 and i < len(book):
            if book[i] > 0:
                take = min(book[i], remaining)
                book[i] -= take
                remaining -= take
                fills.append((i, take))
            i += 1
        return fills

    def snapshot(self, n_levels=5):
        bt = self.touch_index('bid')
        at = self.touch_index('ask')
        bid_prices = self.mid0 - self.dp * (np.arange(n_levels) + (bt if bt is not None else 0) + 1)
        ask_prices = self.mid0 + self.dp * (np.arange(n_levels) + (at if at is not None else 0) + 1)
        # Clip: the touch can drift close enough to the domain edge that
        # touch_index + n_levels runs past the array (a real, if rare, edge
        # case -- pad with 0 rather than raising).
        bid_sizes = np.array([self.bid_book[(bt or 0) + k] if bt is not None and (bt or 0) + k < self.n_ticks else 0.0
                               for k in range(n_levels)])
        ask_sizes = np.array([self.ask_book[(at or 0) + k] if at is not None and (at or 0) + k < self.n_ticks else 0.0
                               for k in range(n_levels)])
        spread = (ask_prices[0] - bid_prices[0]) if bt is not None and at is not None else None
        mid = (ask_prices[0] + bid_prices[0]) / 2.0 if spread is not None else self.mid0
        return {'bid_prices': bid_prices, 'bid_sizes': bid_sizes,
                'ask_prices': ask_prices, 'ask_sizes': ask_sizes,
                'mid': mid, 'spread': spread}


def load_qr_params(path='calibrated_params_7day.json'):
    with open(path) as f:
        d = json.load(f)
    return {k: d[k] for k in ('Q0', 'a_LO', 'b_LO', 'a_C', 'a_MO')}
