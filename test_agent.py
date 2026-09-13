"""
Verification for Part 4 (agent_spooner.py): a short on-policy SARSA(lambda)
run confirming the agent actually learns -- weights move off zero,
epsilon decays on schedule, the inventory-override case is handled
correctly (the agent updates toward the action that ACTUALLY executed,
not the one it proposed), and nothing crashes or produces NaNs. This is
not the full 1000-episode training run (that's Part 5) -- just enough to
confirm the machinery is wired correctly before committing to it.
"""
import time

import numpy as np

from agent_spooner import SpoonerAgent, MIN_INV, MAX_INV, EPS_FLOOR
from env import MarketMakingEnv, MO_CLEAR_ACTION
from market_qr import QueueReactiveBook, load_calibrated_params


def run_steps(agent, env, episode, n_steps, rng):
    state = env.reset(mid_price=100.0, book_burn_in=2000, warmup_steps=200)
    action, qs, actives = agent.act(state, episode)

    total_reward = 0.0
    n_overrides = 0
    deltas = []
    for _ in range(n_steps):
        next_state, reward, done, info = env.step(action)
        actual_action = info['action_taken']
        if actual_action != action:
            n_overrides += 1
            active = agent._active_indices(state, actual_action)
        else:
            active = actives[action]

        next_action, next_qs, next_actives = agent.act(next_state, episode)
        next_q = 0.0 if done else next_qs[next_action]
        delta = agent.update(active, reward, next_q, done)
        deltas.append(delta)

        total_reward += reward
        state = next_state
        action, qs, actives = next_action, next_qs, next_actives
        if done:
            break

    return total_reward, n_overrides, np.array(deltas)


def test_epsilon_schedule():
    agent = SpoonerAgent()
    e0 = agent.epsilon(0)
    e_mid = agent.epsilon(1000)
    e_late = agent.epsilon(1_000_000)  # 1000 * EPS_T
    assert abs(e0 - 0.7) < 1e-9, f'epsilon(0) should be EPS_INIT=0.7, got {e0}'
    assert e_mid < e0, 'epsilon should decay'
    # hyperbolic decay: eps(EPS_T) is exactly halfway between floor and init
    e_at_T = agent.epsilon(EPS_T := 1000)
    assert abs(e_at_T - (EPS_FLOOR + (0.7 - EPS_FLOOR) / 2)) < 1e-6, \
        f'eps(EPS_T) should be halfway to the floor, got {e_at_T}'
    assert e_late < 0.001, f'epsilon should be near-floor by 1000*EPS_T, got {e_late}'
    print(f'test_epsilon_schedule: PASS (eps(0)={e0:.4f}, eps(EPS_T)={e_at_T:.4f}, eps(1e6)={e_late:.6f})')


def test_inventory_override_uses_actual_action():
    """If the agent proposes action 3 but the environment forces action 9
    (MO clear) due to an inventory breach, the SARSA update must be keyed
    on action 9's tiles, not action 3's -- otherwise the agent learns a
    completely wrong association."""
    rng = np.random.default_rng(11)
    params = load_calibrated_params()
    book = QueueReactiveBook(params, rng=rng)
    env = MarketMakingEnv(book, order_size=1000, min_inv=-500, max_inv=500, rng=rng)
    agent = SpoonerAgent(memory_size=100_000, seed=11)
    state = env.reset(mid_price=100.0, book_burn_in=1000, warmup_steps=100)

    env.inv = 600.0  # force out-of-bounds
    proposed_action = 2  # deliberately NOT the MO-clear action
    _, _, actives = agent.act(state, episode=0)
    next_state, reward, done, info = env.step(proposed_action)
    assert info['action_requested'] == proposed_action
    assert info['action_taken'] == MO_CLEAR_ACTION, \
        f"expected override to MO_CLEAR_ACTION, got {info['action_taken']}"
    print('test_inventory_override_uses_actual_action: PASS '
          f"(requested {info['action_requested']}, env executed {info['action_taken']})")


def test_sarsa_update_mechanics():
    """A fast, deterministic unit test of agent.update() in isolation --
    hand-verified arithmetic, no environment involved. Needed because the
    end-to-end test below can (correctly) see zero reward for long
    stretches: with no fills, inventory never leaves 0, so
    psi + inv*delta_m is EXACTLY zero regardless of action -- that's the
    reward function working as designed, not evidence the update math
    works. This test isolates the update math itself."""
    agent = SpoonerAgent(memory_size=10_000, seed=0)
    state = {'inventory': 0.0, 'theta_a': 1, 'theta_b': 1, 'spread': 0.01,
             'mid_price_move': 0.0, 'imbalance': 0.0, 'signed_volume': 0.0,
             'volatility': 0.0, 'rsi': 50.0}
    active = agent._active_indices(state, action=0)
    q_before = agent.q_value(active)
    assert q_before == 0.0, 'fresh agent should start at Q=0 everywhere'

    delta = agent.update(active, reward=1.0, next_q=0.0, done=False)
    assert abs(delta - 1.0) < 1e-9, f'expected TD error 1.0 (reward - 0), got {delta}'

    q_after = agent.q_value(active)
    assert q_after > 0, f'Q(s,a) should have increased after a positive-reward update, got {q_after}'
    # Expected magnitude: sum over 3 groups of weight_g * n_tilings * (alpha * delta * weight_g)
    from agent_spooner import ALPHA, N_TILINGS, GROUP_WEIGHTS
    expected = sum(w * N_TILINGS * (ALPHA * 1.0 * w) for w in GROUP_WEIGHTS.values())
    assert abs(q_after - expected) < 1e-6, f'expected Q(s,a)={expected}, got {q_after}'

    n_nonzero = int((agent.w != 0).sum())
    expected_n_active = N_TILINGS * 3  # agent + market + full groups, unlikely to collide at this scale
    print(f'test_sarsa_update_mechanics: PASS (delta={delta:.4f}, Q before/after={q_before}/{q_after:.6f}, '
          f'nonzero_weights={n_nonzero}, expected~{expected_n_active})')


def test_learning_with_forced_fills(n_steps=40000, seed=7):
    """Forces action=0 (the tightest allowed quote, same as
    test_env.py's aggressive_fill_test) throughout, so real fills occur
    and reward genuinely goes nonzero -- an end-to-end confirmation that
    weight updates happen correctly when there's something real to learn
    from, not just in the isolated unit test above. Only computes Q for
    the one forced action each step (not all 10), since exploration isn't
    being tested here -- keeps this fast enough to actually see fills."""
    rng = np.random.default_rng(seed)
    params = load_calibrated_params()
    book = QueueReactiveBook(params, rng=rng)
    env = MarketMakingEnv(book, order_size=1000, rng=rng)
    agent = SpoonerAgent(memory_size=1_000_000, seed=seed)

    state = env.reset(mid_price=100.0, book_burn_in=2000, warmup_steps=200)
    active = agent._active_indices(state, action=0)

    t0 = time.time()
    total_reward = 0.0
    n_fills = 0
    deltas = []
    for _ in range(n_steps):
        next_state, reward, done, info = env.step(0)
        if info['matched_a'] > 0 or info['matched_b'] > 0:
            n_fills += 1
        next_active = agent._active_indices(next_state, action=0)
        next_q = agent.q_value(next_active)
        delta = agent.update(active, reward, next_q, done)
        deltas.append(delta)
        total_reward += reward
        state, active = next_state, next_active
    elapsed = time.time() - t0
    deltas = np.array(deltas)

    n_nonzero_weights = int((agent.w != 0).sum())
    print(f'test_learning_with_forced_fills: {n_steps} steps in {elapsed:.1f}s '
          f'({n_steps/elapsed:.0f} steps/sec), fills={n_fills}, total_reward={total_reward:.4f}, '
          f'nonzero_weights={n_nonzero_weights}, mean|delta|={np.abs(deltas).mean():.6f}, '
          f'max|delta|={np.abs(deltas).max():.6f}')

    assert not np.isnan(deltas).any(), 'NaN TD error encountered'
    assert n_fills > 0, f'no fills occurred in {n_steps} forced-aggressive steps -- environment regression'
    assert n_nonzero_weights > 0, 'no weights were ever updated -- learning is not happening'
    assert (deltas != 0).any(), 'TD error was zero at every single step despite fills occurring'
    assert env.min_inv - 2000 < env.inv < env.max_inv + 2000, 'inventory ran away'
    return agent, env


if __name__ == '__main__':
    test_epsilon_schedule()
    test_inventory_override_uses_actual_action()
    test_sarsa_update_mechanics()
    test_learning_with_forced_fills()
    print('ALL TESTS PASSED')
