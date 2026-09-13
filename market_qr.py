"""
Queue-reactive limit order book simulator (Part 1 of the RL market-making
plan), driven by the parameters iex_calibrate.py fits from real IEX DEEP
data (calibrated_params_7day.json).

Per iex_calibrate.py's own docstring scope note, only the touch (level 1)
is fit from data; this simulator extends those fitted intensity functions
to all 5 levels per side by reusing them unchanged ("levels 2-5 reuse the
same level-independent intensity functions"). This is a deliberate,
documented simplification, not an attempt to separately fit deeper-level
dynamics we don't have message-level data for.

Event types, each a Poisson process whose rate is a function of the
CURRENT queue size q at that level (the "queue-reactive" property):
  - limit order arrival:  lambda_LO(q) = a_LO * exp(-neg_slope * q)
  - cancellation:          lambda_C(q)  = slope * q
  - market order (touch only, constant rate, split evenly buy/sell):
                            lambda_MO = a_MO

neg_slope = b_LO / Q0 and slope = a_C / Q0 recover the raw per-share fit
constants from the rescaled values iex_calibrate.py reports (see its
calibrate_multi()).

STABILITY NOTE: the 7-day pooled fit has b_LO < 0, so neg_slope < 0 and
lambda_LO(q) is actually INCREASING in q -- a positive-feedback regime
(deeper queue -> faster arrivals -> deeper still) that cancellation's
linear-in-q term eventually can't outpace, since exp(+.) beats a linear
term at large enough q. Confirmed empirically: unclamped, a long enough
run has a random excursion into runaway growth. The fit only has support
over the queue sizes actually observed in the data (~500-ish shares, per
calibrated_params_7day.json's _diagnostics), so extrapolating the
exponential far past that was never statistically justified regardless of
sign -- q is therefore clamped to the observed range before evaluating
either rate function (flat extrapolation beyond the data's support,
rather than trusting an unjustified curve shape out there).
"""
import json
import numpy as np


class QueueReactiveBook:
    def __init__(self, params, n_levels=5, tick_size=0.005, order_size=100,
                 mo_size_sampler=None, q_max_observed=None, rng=None):
        self.Q0 = params['Q0']
        self.a_LO = params['a_LO']
        self.neg_slope = params['b_LO'] / self.Q0
        self.slope = params['a_C'] / self.Q0
        self.a_MO = params['a_MO']
        # Clamp bound: max queue size the fit actually saw. Defaults to
        # whatever load_calibrated_params() attached to `params`, else 450.
        self.q_max_observed = (q_max_observed if q_max_observed is not None
                                else params.get('q_max_observed', 450.0))

        self.n_levels = n_levels
        self.tick_size = tick_size
        self.order_size = order_size
        # MO sizes: draw from the real empirical trade-size distribution if
        # given, else fall back to the fixed chunk size.
        self.mo_size_sampler = mo_size_sampler
        self.rng = rng if rng is not None else np.random.default_rng()

        self.bid_sizes = np.zeros(n_levels)
        self.ask_sizes = np.zeros(n_levels)
        self.bid_price0 = None  # touch bid price
        self.ask_price0 = None  # touch ask price

    def lo_rate(self, q):
        q_capped = np.minimum(q, self.q_max_observed)
        return self.a_LO * np.exp(-self.neg_slope * q_capped)

    def cancel_rate(self, q):
        q_capped = np.minimum(q, self.q_max_observed)
        return self.slope * q_capped

    def reset(self, mid_price=100.0, burn_in_events=3000):
        half_spread_ticks = 1
        self.bid_price0 = mid_price - half_spread_ticks * self.tick_size
        self.ask_price0 = mid_price + half_spread_ticks * self.tick_size
        self.bid_sizes = np.full(self.n_levels, self.Q0, dtype=float)
        self.ask_sizes = np.full(self.n_levels, self.Q0, dtype=float)
        for _ in range(burn_in_events):
            self.step_background()
        return self.snapshot()

    def _rates(self):
        """Per-level, per-side rates for LO arrival and cancellation, plus
        the touch-only MO rate. Returns flat arrays for fast sampling."""
        lo_bid = self.lo_rate(self.bid_sizes)
        lo_ask = self.lo_rate(self.ask_sizes)
        c_bid = np.where(self.bid_sizes > 0, self.cancel_rate(self.bid_sizes), 0.0)
        c_ask = np.where(self.ask_sizes > 0, self.cancel_rate(self.ask_sizes), 0.0)
        mo_buy = self.a_MO / 2.0
        mo_sell = self.a_MO / 2.0
        return lo_bid, lo_ask, c_bid, c_ask, mo_buy, mo_sell

    def step_background(self):
        """Advance the book by exactly one background event. Returns
        (dt, event) where event is a dict describing what happened, for
        the environment layer to use for reward/fill accounting."""
        lo_bid, lo_ask, c_bid, c_ask, mo_buy, mo_sell = self._rates()
        rates = np.concatenate([lo_bid, lo_ask, c_bid, c_ask, [mo_buy, mo_sell]])
        total_rate = rates.sum()
        if total_rate <= 0:
            # Degenerate empty book: force a deposit at the touch to recover.
            self.bid_sizes[0] += self.order_size
            self.ask_sizes[0] += self.order_size
            return 0.0, {'type': 'recover'}

        dt = self.rng.exponential(1.0 / total_rate)
        idx = self.rng.choice(len(rates), p=rates / total_rate)

        n = self.n_levels
        if idx < n:
            level = idx
            self.bid_sizes[level] += self.order_size
            event = {'type': 'LO', 'side': 'bid', 'level': level, 'size': self.order_size}
        elif idx < 2 * n:
            level = idx - n
            self.ask_sizes[level] += self.order_size
            event = {'type': 'LO', 'side': 'ask', 'level': level, 'size': self.order_size}
        elif idx < 3 * n:
            level = idx - 2 * n
            removed = min(self.order_size, self.bid_sizes[level])
            self.bid_sizes[level] -= removed
            self._collapse_if_touch_empty('bid')
            event = {'type': 'cancel', 'side': 'bid', 'level': level, 'size': removed}
        elif idx < 4 * n:
            level = idx - 3 * n
            removed = min(self.order_size, self.ask_sizes[level])
            self.ask_sizes[level] -= removed
            self._collapse_if_touch_empty('ask')
            event = {'type': 'cancel', 'side': 'ask', 'level': level, 'size': removed}
        elif idx == 4 * n:
            size = self._draw_mo_size()
            fills = self.apply_market_order('sell', size)  # buy MO hits the ask
            event = {'type': 'MO', 'side': 'buy', 'size': size, 'fills': fills}
        else:
            size = self._draw_mo_size()
            fills = self.apply_market_order('buy', size)  # sell MO hits the bid
            event = {'type': 'MO', 'side': 'sell', 'size': size, 'fills': fills}

        return dt, event

    def _draw_mo_size(self):
        if self.mo_size_sampler is not None:
            return float(self.mo_size_sampler(self.rng))
        return float(self.order_size)

    def _collapse_if_touch_empty(self, side):
        """Shift the level array so the touch is non-empty again, promoting
        deeper levels up. Bounded to n_levels-1 shifts, since shifting more
        than that is meaningless (the array only has n_levels slots) --
        an unbounded `while` here would spin forever if every level on this
        side is simultaneously empty (observed in practice: possible after
        a chain of cancellations, not just a theoretical edge case)."""
        sizes = self.bid_sizes if side == 'bid' else self.ask_sizes
        shifts = 0
        while sizes[0] <= 0 and shifts < self.n_levels - 1:
            sizes[:-1] = sizes[1:]
            sizes[-1] = 0.0
            if side == 'bid':
                self.bid_price0 -= self.tick_size
            else:
                self.ask_price0 += self.tick_size
            shifts += 1
        if sizes[0] <= 0:
            # Entire side emptied out -- reseed the touch with one chunk
            # rather than leaving a degenerate all-zero book.
            sizes[0] = self.order_size

    def apply_market_order(self, resting_side, size):
        """A market order that removes `size` shares from the book side
        `resting_side` ('bid' or 'sell-side liquidity', 'ask' likewise),
        walking through levels on a partial fill. Returns a list of
        (level, filled_qty) tuples."""
        sizes = self.bid_sizes if resting_side == 'bid' else self.ask_sizes
        fills = []
        remaining = size
        level = 0
        while remaining > 0 and level < self.n_levels:
            avail = sizes[level]
            take = min(avail, remaining)
            if take > 0:
                sizes[level] -= take
                remaining -= take
                fills.append((level, take))
            if sizes[level] <= 0:
                self._collapse_if_touch_empty(resting_side)
                level = 0  # collapsed; re-check from the (new) touch
            else:
                break
        return fills

    def snapshot(self):
        bid_prices = self.bid_price0 - self.tick_size * np.arange(self.n_levels)
        ask_prices = self.ask_price0 + self.tick_size * np.arange(self.n_levels)
        return {
            'bid_prices': bid_prices, 'bid_sizes': self.bid_sizes.copy(),
            'ask_prices': ask_prices, 'ask_sizes': self.ask_sizes.copy(),
            'mid': (bid_prices[0] + ask_prices[0]) / 2.0,
            'spread': ask_prices[0] - bid_prices[0],
        }


def load_calibrated_params(path='calibrated_params_7day.json'):
    with open(path) as f:
        d = json.load(f)
    params = {k: d[k] for k in ('Q0', 'a_LO', 'b_LO', 'a_C', 'a_MO')}
    diag = d.get('_diagnostics', {})
    q_maxes = []
    if diag.get('arrival_fit'):
        q_maxes.append(max(diag['arrival_fit']['q_mid']))
    if diag.get('cancel_fit'):
        q_maxes.append(max(diag['cancel_fit']['q_mid']))
    params['q_max_observed'] = max(q_maxes) if q_maxes else 450.0
    return params


def load_trade_size_sampler(archive_dir='archive', symbol='INTC'):
    """Empirical MO size distribution from real trade data, for the
    queue-reactive simulator's market orders (kept realistic/heavy-tailed,
    unlike the fixed-chunk assumption used in the Poisson control)."""
    import csv
    import glob
    import os
    sizes = []
    for prl_dir in sorted(glob.glob(os.path.join(archive_dir, '*'))):
        trd_files = glob.glob(os.path.join(prl_dir, '*_trd.csv'))
        if not trd_files:
            continue
        with open(trd_files[0]) as f:
            for row in csv.DictReader(f):
                if row['Symbol'] == symbol:
                    sizes.append(float(row['Size']))
    sizes = np.array(sizes)

    def sampler(rng):
        return rng.choice(sizes)

    return sampler
