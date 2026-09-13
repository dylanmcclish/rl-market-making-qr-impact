"""
Calibrate market.py's queue-reactive parameters from real IEX DEEP data,
via the third-party `iex-cppparser` package (pip install iex-cppparser),
which downloads+parses IEX's free HIST archive into two CSVs per day:

  _prl.csv (price level updates), columns (no header):
    packet_capture_ns, send_time_ns, exch_timestamp_ns, "PRL", symbol,
    price, size, record_type ('Z' if size==0 else 'R'), event_flag (0/1),
    side (0=bid, 1=ask)

  _trd.csv (trade reports), columns (no header):
    packet_capture_ns, send_time_ns, exch_timestamp_ns, "T", symbol,
    size, price, trade_id, sale_condition_string

IMPORTANT DIFFERENCE FROM LOBSTER: IEX's price-level-update messages report
the *absolute new displayed size* at a price after some change, not a typed
event (submission/cancel/execution). To classify a decrease as a cancellation
vs. an execution, we cross-reference the trade-report stream: a size decrease
at a price with a same-price, same-time trade report is attributed to an
execution; otherwise it's attributed to a cancellation. This is a standard
necessary approximation when working from a quote/depth feed rather than a
full order-message feed (LOBSTER has the latter; IEX DEEP has the former) --
call it out explicitly in any write-up that uses this.

SCOPE: this calibrates the TOUCH (level 1) only, matching how market.py's
level-1 fit is used as the headline a_LO/b_LO/a_C/a_MO parameters (levels
2-5 in market.py reuse the same level-independent intensity functions, so a
level-1-only calibration is what actually matters for the experiment).

iex-cppparser is a third-party, unofficial community library -- sanity-check
its first day of real output (e.g. rough daily volume, plausible price
range) before trusting it for calibration.

Usage:
    python iex_calibrate.py --prl SYMBOL_prl.csv --trd SYMBOL_trd.csv \
        --symbol INTC --out calibrated_params.json
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# Column order as actually written by the installed iex-cppparser (its CSVs
# carry their own header row, and put the bid/ask side flag right after
# send time rather than last -- both differ from this script's original
# assumption, so the file's header is skipped and columns are relabeled here
# rather than trusted positionally as written).
PRL_COLS = ['cap_ns', 'send_ns', 'side', 'exch_ns', 'tag', 'symbol', 'price',
            'size', 'record_type', 'event_flag']
TRD_COLS = ['cap_ns', 'send_ns', 'exch_ns', 'tag', 'symbol', 'size', 'price',
            'trade_id', 'sale_cond']

TRADE_MATCH_TOLERANCE_NS = 2000  # cross-reference window for execution vs. cancel


def load_files(prl_path, trd_path, symbol):
    prl = pd.read_csv(prl_path, header=0, names=PRL_COLS)
    trd = pd.read_csv(trd_path, header=0, names=TRD_COLS)
    prl = prl[prl['symbol'] == symbol].sort_values('exch_ns').reset_index(drop=True)
    trd = trd[trd['symbol'] == symbol].sort_values('exch_ns').reset_index(drop=True)
    return prl, trd


def reconstruct_touch_events(prl, trd):
    """Walk the price-level-update stream in time order, tracking only the
    best bid/ask (touch), and emit (side, event_kind, q_pre, dt) tuples for
    the binned-intensity fit, plus raw MO inter-arrival times."""
    # index trades by (price) for quick nearby-time lookup
    trd_by_price = {}
    for _, r in trd.iterrows():
        trd_by_price.setdefault(round(r['price'], 4), []).append(r['exch_ns'])
    for k in trd_by_price:
        trd_by_price[k] = np.array(sorted(trd_by_price[k]))

    def trade_matches(price, t_ns):
        arr = trd_by_price.get(round(price, 4))
        if arr is None or len(arr) == 0:
            return False
        idx = np.searchsorted(arr, t_ns)
        for j in (idx - 1, idx):
            if 0 <= j < len(arr) and abs(int(arr[j]) - int(t_ns)) <= TRADE_MATCH_TOLERANCE_NS:
                return True
        return False

    books = {'bid': {}, 'ask': {}}          # price -> size
    touch = {'bid': (None, None), 'ask': (None, None)}   # (price, size)
    last_touch_change_ns = {'bid': None, 'ask': None}

    lo_pairs = {'bid': [], 'ask': []}   # (q_pre, dt)
    c_pairs = {'bid': [], 'ask': []}
    mo_events_ns = {'bid': [], 'ask': []}

    def best_price(side):
        d = books[side]
        if not d:
            return None
        return max(d) if side == 'bid' else min(d)

    for _, row in prl.iterrows():
        side = 'bid' if row['side'] == 0 else 'ask'
        price = row['price']
        new_size = row['size']
        t_ns = row['exch_ns']

        old_size = books[side].get(price, 0.0)
        pre_touch_price, pre_touch_size = touch[side]
        is_touch_event = (pre_touch_price is not None and price == pre_touch_price)

        # apply update to book state
        if new_size <= 0:
            books[side].pop(price, None)
        else:
            books[side][price] = new_size

        if is_touch_event:
            delta = new_size - old_size
            if last_touch_change_ns[side] is not None:
                dt = (t_ns - last_touch_change_ns[side]) / 1e9  # ns -> s
                if dt > 0 and pre_touch_size is not None:
                    if delta > 0:
                        lo_pairs[side].append((pre_touch_size, dt))
                    elif delta < 0:
                        if trade_matches(price, t_ns):
                            mo_events_ns[side].append(t_ns)
                        else:
                            c_pairs[side].append((pre_touch_size, dt))
            last_touch_change_ns[side] = t_ns

        # recompute touch (handles promotions when the old touch emptied out)
        bp = best_price(side)
        touch[side] = (bp, books[side].get(bp) if bp is not None else None)
        if not is_touch_event and touch[side][0] != pre_touch_price:
            # the touch moved to a different price (promotion); reset the
            # sojourn clock at the new touch without counting it as an
            # arrival event, matching market.py's `_promote` semantics
            last_touch_change_ns[side] = t_ns

    return lo_pairs, c_pairs, mo_events_ns


# --- same binned-MLE fitting logic as calibrate.py (LOBSTER version) -------
def fit_arrival(pairs, n_bins=15):
    if len(pairs) < n_bins * 5:
        return None
    q = np.array([p[0] for p in pairs]); dt = np.array([p[1] for p in pairs])
    edges = np.unique(np.quantile(q, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    bin_idx = np.digitize(q, edges[1:-1])
    q_mid, rates = [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        total_t = dt[mask].sum()
        if total_t <= 0 or mask.sum() < 5:
            continue
        q_mid.append(q[mask].mean()); rates.append(mask.sum() / total_t)
    if len(q_mid) < 3:
        return None
    q_mid = np.array(q_mid); rates = np.array(rates)
    valid = rates > 0
    logr = np.log(rates[valid])
    A = np.vstack([np.ones(valid.sum()), q_mid[valid]]).T
    coef, *_ = np.linalg.lstsq(A, logr, rcond=None)
    log_a, slope = coef
    return {'a': float(np.exp(log_a)), 'neg_slope': float(-slope),
            'q_mid': q_mid.tolist(), 'rates': rates.tolist()}


def fit_cancel(pairs, n_bins=15):
    if len(pairs) < n_bins * 5:
        return None
    q = np.array([p[0] for p in pairs]); dt = np.array([p[1] for p in pairs])
    edges = np.unique(np.quantile(q, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    bin_idx = np.digitize(q, edges[1:-1])
    q_mid, rates = [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        total_t = dt[mask].sum()
        if total_t <= 0 or mask.sum() < 5:
            continue
        q_mid.append(q[mask].mean()); rates.append(mask.sum() / total_t)
    if len(q_mid) < 3:
        return None
    q_mid = np.array(q_mid); rates = np.array(rates)
    slope = float(np.sum(q_mid * rates) / np.sum(q_mid ** 2))
    return {'slope': slope, 'q_mid': q_mid.tolist(), 'rates': rates.tolist()}


def calibrate_multi(day_files, symbol):
    """Pool touch events across multiple trading days and fit one set of
    parameters from the combined sample. day_files is a list of
    (prl_path, trd_path) pairs -- one per day. Each day's events are
    reconstructed independently (a session boundary is never bridged into
    a fake overnight sojourn), then LO/cancel observations are pooled
    directly (each carries its own dt, so pooling is exact regardless of
    which day it came from) and MO's constant-rate estimate is built from
    the SUM of each day's own (count, span) rather than the span between
    the first and last trade across all days, which would otherwise be
    inflated by the overnight/weekend gaps between sessions."""
    lo_pooled, c_pooled = [], []
    mo_total_count = 0
    mo_total_span_s = 0.0
    touch_sizes = []
    n_days = 0

    for prl_path, trd_path in day_files:
        prl, trd = load_files(prl_path, trd_path, symbol)
        if len(prl) == 0 and len(trd) == 0:
            continue
        n_days += 1
        lo_pairs, c_pairs, mo_ns = reconstruct_touch_events(prl, trd)

        for side in ('bid', 'ask'):
            touch_sizes += [p[0] for p in lo_pairs[side]] + [p[0] for p in c_pairs[side]]
        lo_pooled += lo_pairs['bid'] + lo_pairs['ask']
        c_pooled += c_pairs['bid'] + c_pairs['ask']

        day_mo = sorted(mo_ns['bid'] + mo_ns['ask'])
        if len(day_mo) > 1:
            span_s = (day_mo[-1] - day_mo[0]) / 1e9
            if span_s > 0:
                mo_total_count += len(day_mo)
                mo_total_span_s += span_s

    Q0 = float(np.median(touch_sizes)) if touch_sizes else None
    arr_fit = fit_arrival(lo_pooled)
    can_fit = fit_cancel(c_pooled)
    a_MO = float(mo_total_count / mo_total_span_s) if mo_total_span_s > 0 else None

    result = {'symbol': symbol, 'n_days': n_days, 'Q0': Q0}
    if arr_fit and Q0:
        result['a_LO'] = arr_fit['a']
        result['b_LO'] = arr_fit['neg_slope'] * Q0
    if can_fit and Q0:
        result['a_C'] = can_fit['slope'] * Q0
    if a_MO is not None:
        result['a_MO'] = a_MO
    result['n_LO_events'] = len(lo_pooled)
    result['n_C_events'] = len(c_pooled)
    result['n_MO_events'] = mo_total_count
    result['_diagnostics'] = {'arrival_fit': arr_fit, 'cancel_fit': can_fit}
    return result


def calibrate(prl_path, trd_path, symbol):
    return calibrate_multi([(prl_path, trd_path)], symbol)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prl', help='single day: path to one SYMBOL_prl.csv')
    p.add_argument('--trd', help='single day: path to one SYMBOL_trd.csv')
    p.add_argument('--archive-dir', help='multi-day: pool every dated subfolder\'s '
                    '*_prl.csv/*_trd.csv pair under this archive directory')
    p.add_argument('--symbol', required=True)
    p.add_argument('--out', default='calibrated_params.json')
    args = p.parse_args()

    if args.archive_dir:
        import glob
        day_files = []
        for prl_path in sorted(glob.glob(os.path.join(args.archive_dir, '*', '*_prl.csv'))):
            day_dir = os.path.dirname(prl_path)
            trd_candidates = glob.glob(os.path.join(day_dir, '*_trd.csv'))
            if trd_candidates:
                day_files.append((prl_path, trd_candidates[0]))
        if not day_files:
            sys.exit(f"No *_prl.csv/*_trd.csv pairs found under {args.archive_dir}")
        res = calibrate_multi(day_files, args.symbol)
    elif args.prl and args.trd:
        res = calibrate(args.prl, args.trd, args.symbol)
    else:
        sys.exit("Pass either --prl/--trd (one day) or --archive-dir (pool all days)")

    print(json.dumps({k: v for k, v in res.items() if k != '_diagnostics'}, indent=2), file=sys.stderr)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved {args.out}", file=sys.stderr)
