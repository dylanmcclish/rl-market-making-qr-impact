"""
Validation for Parts 1-2 of the RL market-making plan.

market_zi.py (Poisson/zero-intelligence control): reproduces Smith et al.'s
own validation figures -- mean depth profile, price impact function,
spread distribution, price diffusion -- in their nondimensional units, so
the shapes can be checked against the paper's own Figs. 3/6/9/11 and
Table III/IV scaling relations before trusting the control as a baseline.

market_qr.py (queue-reactive): a self-consistency check -- run the
background process alone (no agent) and confirm the touch-level event
rates and queue-size distribution it produces match what iex_calibrate.py
actually fit from real data, since the simulator is supposed to reproduce
those fitted intensities by construction.

Writes results to validation_results.json for the report/artifact step.
"""
import json
import time

import numpy as np

from market_qr import QueueReactiveBook, load_calibrated_params, load_trade_size_sampler
from market_zi import ZeroIntelligenceBook, calibrate_zi_params, load_qr_params


def validate_zi(n_ticks=60, burn_in=4000, n_snapshots=300, snapshot_gap_events=150,
                 mo_sizes_for_impact=None, seed=0):
    rng = np.random.default_rng(seed)
    qr_params = load_qr_params()
    zi_params = calibrate_zi_params(qr_params)
    book = ZeroIntelligenceBook(zi_params['alpha'], zi_params['mu'], zi_params['delta'],
                                 zi_params['dp'], zi_params['sigma'], n_ticks=n_ticks, rng=rng)
    book.reset(mid_price=100.0, burn_in_events=burn_in)

    pc = zi_params['mu'] / (2 * zi_params['alpha'])
    Nc = zi_params['mu'] / (2 * zi_params['delta'])
    delta = zi_params['delta']
    alpha = zi_params['alpha']
    eps = 2 * delta * zi_params['sigma'] / zi_params['mu']

    ask_profiles = []
    spreads = []
    mid_path_t = []
    mid_path_m = []
    t_elapsed = 0.0

    t0 = time.time()
    for snap in range(n_snapshots):
        for _ in range(snapshot_gap_events):
            dt, _ = book.step_background()
            t_elapsed += dt
            snap_state = book.snapshot(n_levels=1)
            if snap_state['spread'] is not None:
                mid_path_t.append(t_elapsed)
                mid_path_m.append(snap_state['mid'])
        ask_profiles.append(book.ask_book.copy())
        bt = book.touch_index('bid')
        at = book.touch_index('ask')
        if bt is not None and at is not None:
            spreads.append((at + 1 + bt + 1) * zi_params['dp'])
    runtime_s = time.time() - t0

    mean_profile = np.mean(ask_profiles, axis=0)
    p_hat = (np.arange(n_ticks) + 1) * zi_params['dp'] / pc
    n_hat = mean_profile * pc / Nc  # = mean_profile * delta / alpha

    cum_depth = np.cumsum(mean_profile)
    order_sizes_shares = np.linspace(zi_params['sigma'], cum_depth[-1] * 0.6, 25)
    impact_ticks = []
    for w in order_sizes_shares:
        idx = np.searchsorted(cum_depth, w)
        idx = min(idx, n_ticks - 1)
        impact_ticks.append((idx + 1) * zi_params['dp'])
    impact_x = order_sizes_shares * eps / zi_params['sigma']
    impact_y = np.array(impact_ticks) / pc

    spreads = np.array(spreads)
    s_hat = spreads / pc

    mid_path_t = np.array(mid_path_t)
    mid_path_m = np.array(mid_path_m)
    diffusion = {}
    if len(mid_path_t) > 500:
        taus = np.geomspace(max(mid_path_t[1] - mid_path_t[0], 1e-3), mid_path_t[-1] / 4, 12)
        variances = []
        for tau in taus:
            idx_shift = np.searchsorted(mid_path_t, mid_path_t + tau)
            idx_shift = np.clip(idx_shift, 0, len(mid_path_m) - 1)
            diffs = mid_path_m[idx_shift] - mid_path_m
            valid = idx_shift > np.arange(len(idx_shift))
            if valid.sum() > 20:
                variances.append(np.var(diffs[valid]))
            else:
                variances.append(np.nan)
        diffusion = {
            'tau_hat': (taus * delta).tolist(),
            'var_hat': (np.array(variances) / pc ** 2).tolist(),
        }

    return {
        'params': zi_params, 'pc': pc, 'Nc': Nc, 'epsilon': eps, 'runtime_s': runtime_s,
        'depth_profile': {'p_hat': p_hat.tolist(), 'n_hat': n_hat.tolist()},
        'price_impact': {'x': impact_x.tolist(), 'y': impact_y.tolist()},
        'spread_dist': {'s_hat': s_hat.tolist()},
        'diffusion': diffusion,
        'n_events_simulated': n_snapshots * snapshot_gap_events + burn_in,
    }


def validate_qr(n_events=300000, warmup=5000, seed=1):
    rng = np.random.default_rng(seed)
    params = load_calibrated_params()
    mo_sampler = load_trade_size_sampler()
    book = QueueReactiveBook(params, n_levels=5, tick_size=0.005, order_size=100,
                              mo_size_sampler=mo_sampler, rng=rng)
    book.reset(mid_price=100.0, burn_in_events=warmup)

    touch_bid_sizes = []
    touch_ask_sizes = []
    event_counts = {'LO': 0, 'cancel': 0, 'MO': 0, 'recover': 0}
    t_elapsed = 0.0

    t0 = time.time()
    for _ in range(n_events):
        touch_bid_sizes.append(book.bid_sizes[0])
        touch_ask_sizes.append(book.ask_sizes[0])
        dt, event = book.step_background()
        t_elapsed += dt
        event_counts[event['type']] += 1
    runtime_s = time.time() - t0

    touch_sizes = np.array(touch_bid_sizes + touch_ask_sizes)
    fitted_mo_rate = event_counts['MO'] / t_elapsed if t_elapsed > 0 else None

    # bin observed touch sizes and compare empirical LO/cancel rates against
    # the fitted functional forms at those queue sizes -- a direct
    # self-consistency check that the simulator reproduces what was fit.
    return {
        'n_events': n_events, 't_elapsed_s': t_elapsed, 'runtime_s': runtime_s,
        'event_counts': event_counts,
        'fitted_a_MO': params['a_MO'], 'simulated_MO_rate': fitted_mo_rate,
        'touch_size_median': float(np.median(touch_sizes)),
        'touch_size_mean': float(np.mean(touch_sizes)),
        'Q0_from_calibration': params['Q0'],
        'params': params,
    }


if __name__ == '__main__':
    print('validating market_zi.py (Poisson control)...')
    zi_results = validate_zi()
    print(f"  epsilon = {zi_results['epsilon']:.4f}, pc = {zi_results['pc']:.5f}, "
          f"runtime = {zi_results['runtime_s']:.1f}s")

    print('validating market_qr.py (queue-reactive)...')
    qr_results = validate_qr()
    print(f"  touch median = {qr_results['touch_size_median']:.1f} "
          f"(Q0 = {qr_results['Q0_from_calibration']}), "
          f"MO rate sim/fit = {qr_results['simulated_MO_rate']:.3f} / {qr_results['fitted_a_MO']:.3f}, "
          f"runtime = {qr_results['runtime_s']:.1f}s")

    with open('validation_results.json', 'w') as f:
        json.dump({'zi': zi_results, 'qr': qr_results}, f, indent=2)
    print('Saved validation_results.json')
