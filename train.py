"""
Part 5: training and evaluation.

Training is Hogwild-style, deliberately mirroring how the actual
tspooner/rl_markets C++ agent made 1000-episode training tractable (see
main.cpp's `train()`/`run()`): multiple OS processes each run independent
full-day episodes concurrently against ONE shared weight vector, with NO
locking around the learning update itself -- only a shared episode counter
is synchronized, exactly like their `episode_mutex`. Asynchronous,
occasionally-stale updates are the accepted price for wall-clock that
scales close to linearly with core count; this is the same trade-off
Hogwild! SGD and A3C make. Everything else -- algorithm, hyperparameters,
action space, reward, episode definition (one simulated trading day, full
event granularity, no coarser decision cadence) -- is unchanged from
agent_spooner.py / env.py.

`multiprocessing` (real OS processes, not threads) is required: Python
threads stay GIL-bound for the per-step control flow here, so they would
not deliver the real parallelism the C++ agent gets from OS threads over
un-synchronized shared memory.
"""
import argparse
import ctypes
import csv
import multiprocessing as mp
import os
import time

import numpy as np

from agent_spooner import (SpoonerAgent, MEMORY_SIZE, N_TILINGS, TILES_PER_DIM,
                            MIN_INV, MAX_INV, ORDER_SIZE)
from env import MarketMakingEnv, N_ACTIONS, MO_CLEAR_ACTION, ACTION_THETA
from market_qr import QueueReactiveBook, load_calibrated_params, load_trade_size_sampler
from market_zi import ZeroIntelligenceBook, calibrate_zi_params

# Empirically measured (see the derivation this session ran against the
# calibrated QR simulator): mean background-event dt ~1.328ms -> a 6.5-hour
# trading day is ~17.6M events. This is what "one episode" means throughout
# -- a full simulated day, matching the paper's day-granularity episodes,
# not a shortened proxy.
EVENTS_PER_DAY = 17_600_000

TILE_SEED = 0  # fixed across every worker so hashed feature indices agree


def make_qr_book(rng):
    params = load_calibrated_params()
    sampler = load_trade_size_sampler()
    return QueueReactiveBook(params, mo_size_sampler=sampler, rng=rng)


def make_zi_book(rng):
    qr_params = load_calibrated_params()
    zi_p = calibrate_zi_params(qr_params)
    return ZeroIntelligenceBook(zi_p['alpha'], zi_p['mu'], zi_p['delta'],
                                 zi_p['dp'], zi_p['sigma'], n_ticks=200, rng=rng)


BOOK_FACTORIES = {'qr': make_qr_book, 'zi': make_zi_book}


# ---------------------------------------------------------------------- #
# One on-policy SARSA(lambda) episode against a shared (possibly
# shared-memory) agent -- used by every training worker.
# ---------------------------------------------------------------------- #
def run_training_episode(agent, env, episode_idx, n_events, log_every=None):
    state = env.reset(mid_price=100.0, book_burn_in=3000, warmup_steps=200)
    action, qs, actives = agent.act(state, episode_idx)

    total_reward = 0.0
    n_fills = 0
    abs_inv_sum = 0.0

    for t in range(n_events):
        next_state, reward, done, info = env.step(action)
        actual_action = info['action_taken']
        active = actives[actual_action] if actual_action == action \
            else agent._active_indices(state, actual_action)

        next_action, next_qs, next_actives = agent.act(next_state, episode_idx)
        next_q = next_qs[next_action]
        agent.update(active, reward, next_q, done)

        if info['matched_a'] > 0 or info['matched_b'] > 0:
            n_fills += 1
        total_reward += reward
        abs_inv_sum += abs(env.inv)

        state = next_state
        action, qs, actives = next_action, next_qs, next_actives

        if log_every and (t + 1) % log_every == 0:
            print(f'  [ep {episode_idx}] {t + 1}/{n_events} events, '
                  f'reward={total_reward:.2f}, fills={n_fills}, inv={env.inv:.0f}')

    agent.traces.clear()
    pnl = env.cash + env.inv * env.book.snapshot()['mid']
    return {
        'episode': episode_idx,
        'reward': total_reward,
        'pnl': pnl,
        'n_events': n_events,
        'n_fills': n_fills,
        'mean_abs_inv': abs_inv_sum / n_events,
        'final_inv': env.inv,
    }


# ---------------------------------------------------------------------- #
# Hogwild worker: claims episode indices from a shared counter (the ONLY
# synchronized state besides theta itself, matching main.cpp's
# episode_mutex), runs each one against the shared weight buffer, and
# reports per-episode stats back to the main process via a Queue.
# ---------------------------------------------------------------------- #
def _worker(worker_id, theta_buf, memory_size, n_total_episodes, episode_counter,
            counter_lock, n_events, book_kind, base_seed, result_queue, log_every):
    rng = np.random.default_rng(base_seed + 1000 * worker_id)
    agent = SpoonerAgent(memory_size=memory_size, n_tilings=N_TILINGS,
                          tiles_per_dim=TILES_PER_DIM, seed=TILE_SEED,
                          theta_buffer=theta_buf)
    agent.rng = np.random.default_rng(base_seed + 1000 * worker_id + 1)
    book_factory = BOOK_FACTORIES[book_kind]

    while True:
        with counter_lock:
            if episode_counter.value >= n_total_episodes:
                break
            episode_idx = episode_counter.value
            episode_counter.value += 1

        episode_rng = np.random.default_rng(base_seed + 7919 * (episode_idx + 1))
        book = book_factory(episode_rng)
        env = MarketMakingEnv(book, order_size=ORDER_SIZE, min_inv=MIN_INV,
                               max_inv=MAX_INV, rng=episode_rng)

        t0 = time.time()
        stats = run_training_episode(agent, env, episode_idx, n_events, log_every)
        stats['worker'] = worker_id
        stats['wall_seconds'] = time.time() - t0
        result_queue.put(stats)


def train_hogwild(n_episodes, n_workers, n_events=EVENTS_PER_DAY, book_kind='qr',
                   memory_size=MEMORY_SIZE, base_seed=0, log_every=None,
                   out_dir='.'):
    theta_buf = mp.RawArray(ctypes.c_float, memory_size)  # shared, zero-init by default

    episode_counter = mp.Value('i', 0)
    counter_lock = mp.Lock()
    result_queue = mp.Queue()

    procs = []
    for w in range(n_workers):
        p = mp.Process(target=_worker, args=(w, theta_buf, memory_size, n_episodes,
                                              episode_counter, counter_lock, n_events,
                                              book_kind, base_seed, result_queue, log_every))
        p.start()
        procs.append(p)

    log_path = os.path.join(out_dir, f'training_log_{book_kind}.csv')
    fieldnames = ['episode', 'worker', 'reward', 'pnl', 'n_events', 'n_fills',
                  'mean_abs_inv', 'final_inv', 'wall_seconds']
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        n_done = 0
        t_start = time.time()
        while n_done < n_episodes:
            stats = result_queue.get()
            writer.writerow({k: stats[k] for k in fieldnames})
            f.flush()
            n_done += 1
            elapsed = time.time() - t_start
            print(f'[{n_done}/{n_episodes}] worker={stats["worker"]} '
                  f'reward={stats["reward"]:.2f} pnl={stats["pnl"]:.2f} '
                  f'fills={stats["n_fills"]} wall={stats["wall_seconds"]:.1f}s '
                  f'(elapsed {elapsed / 60:.1f} min)')

    for p in procs:
        p.join()

    theta = np.frombuffer(theta_buf, dtype=np.float32).copy()
    weights_path = os.path.join(out_dir, f'trained_weights_{book_kind}.npy')
    np.save(weights_path, theta)
    print(f'Saved trained weights -> {weights_path}')
    return theta


# ---------------------------------------------------------------------- #
# Evaluation: greedy consolidated agent, plus Spooner's own benchmark
# policies (fixed theta_a=theta_b spread, random), against both
# simulators, using the paper's own metrics (normalized daily PnL, mean
# absolute position).
# ---------------------------------------------------------------------- #
def greedy_policy_fn(agent):
    def policy(state, rng):
        qs, _ = agent.q_all_actions(state)
        return int(np.argmax(qs))
    return policy


def fixed_spread_policy_fn(theta):
    action = ACTION_THETA.index((theta, theta))

    def policy(state, rng):
        return action
    return policy


def random_policy_fn():
    def policy(state, rng):
        return int(rng.integers(0, N_ACTIONS))
    return policy


def evaluate_policy(policy_fn, book_kind, n_episodes, n_events, base_seed=10_000):
    book_factory = BOOK_FACTORIES[book_kind]
    results = []
    for i in range(n_episodes):
        rng = np.random.default_rng(base_seed + 7919 * (i + 1))
        book = book_factory(rng)
        env = MarketMakingEnv(book, order_size=ORDER_SIZE, min_inv=MIN_INV,
                               max_inv=MAX_INV, rng=rng)
        state = env.reset(mid_price=100.0, book_burn_in=3000, warmup_steps=200)
        abs_inv_sum = 0.0
        for t in range(n_events):
            action = policy_fn(state, rng)
            state, reward, done, info = env.step(action)
            abs_inv_sum += abs(env.inv)
        pnl = env.cash + env.inv * env.book.snapshot()['mid']
        results.append({'episode': i, 'pnl': pnl, 'norm_pnl': pnl / n_events,
                         'mean_abs_inv': abs_inv_sum / n_events})
    pnls = np.array([r['norm_pnl'] for r in results])
    maps = np.array([r['mean_abs_inv'] for r in results])
    return {'norm_pnl_mean': float(pnls.mean()), 'norm_pnl_std': float(pnls.std()),
            'map_mean': float(maps.mean()), 'per_episode': results}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'smoke'], default='smoke')
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument('--book', choices=['qr', 'zi'], default='qr')
    ap.add_argument('--events', type=int, default=EVENTS_PER_DAY)
    args = ap.parse_args()

    if args.mode == 'smoke':
        # Small, fast: a handful of short "episodes" per worker, just to
        # confirm the shared-memory Hogwild machinery actually works
        # (weights move, no crashes) and to measure real multi-process
        # scaling before committing to a multi-day run.
        train_hogwild(n_episodes=args.workers * 3, n_workers=args.workers,
                      n_events=4000, book_kind=args.book, log_every=None)
    else:
        train_hogwild(n_episodes=args.episodes, n_workers=args.workers,
                      n_events=args.events, book_kind=args.book)
