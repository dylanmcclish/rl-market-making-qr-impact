"""
Verification for Part 3 (env.py), per the plan's verification bullet:
hand-computed fill scenarios first, then a broader smoke test running the
full random-policy loop against both simulators to confirm nothing
crashes and inventory/reward statistics look sane.
"""
import numpy as np

from env import MarketMakingEnv


class FakeBook:
    """A minimal stand-in with a fixed, hand-set book state, so fill logic
    can be checked against a known-by-hand answer instead of a live
    simulator's stochastic output."""
    def __init__(self):
        self.bid_prices = np.array([99.995, 99.990, 99.985, 99.980, 99.975])
        self.bid_sizes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        self.ask_prices = np.array([100.005, 100.010, 100.015, 100.020, 100.025])
        self.ask_sizes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        self._events = []

    def reset(self, mid_price=100.0, burn_in_events=0):
        return self.snapshot()

    def queue_event(self, event):
        self._events.append(event)

    def step_background(self):
        return 0.01, self._events.pop(0)

    def snapshot(self):
        return {'bid_prices': self.bid_prices.copy(), 'bid_sizes': self.bid_sizes.copy(),
                'ask_prices': self.ask_prices.copy(), 'ask_sizes': self.ask_sizes.copy(),
                'mid': (self.bid_prices[0] + self.ask_prices[0]) / 2,
                'spread': self.ask_prices[0] - self.bid_prices[0]}


def test_market_order_partial_fill():
    """Agent's bid sits at 99.995 (the touch) with 40 shares of real
    volume ahead of it (level shows 100 total; we manually place the
    agent behind an 'ahead' of 40, i.e. as if it joined when 40 were
    already resting -- MarketMakingEnv._new_agent_order would read the
    CURRENT level size as `ahead`, so we drive that path directly).
    A 70-share sell market order arrives: it should eat the 40 ahead,
    then fill 30 of the agent's order, leaving 970 remaining and 0 ahead."""
    book = FakeBook()
    env = MarketMakingEnv(book, order_size=1000, tick_size=0.005)
    env.reset(mid_price=100.0, book_burn_in=0, warmup_steps=0)

    snap = book.snapshot()
    book.bid_sizes[0] = 40.0  # 40 shares "ahead" at the touch before the agent's own order
    env.agent_bid = env._new_agent_order('bid', 99.995, book.snapshot())
    assert env.agent_bid['ahead'] == 40.0
    assert env.agent_bid['remaining'] == 1000.0

    pre_snap = book.snapshot()
    event = {'type': 'MO', 'side': 'sell', 'size': 70.0, 'fills': [(0, 70.0)]}
    matched_a, matched_b, p_a, p_b = env._resolve_fills(pre_snap, event)

    assert matched_b == 30.0, f'expected 30 filled, got {matched_b}'
    assert env.agent_bid['ahead'] == 0.0
    assert env.agent_bid['remaining'] == 970.0
    print('test_market_order_partial_fill: PASS')


def test_cancellation_is_probabilistic_not_a_fill():
    """A cancellation at the agent's price must never directly fill the
    agent (only reduce/leave `ahead`, stochastically) -- unlike a market
    order. Run many trials and check the ahead-reduction rate matches the
    ahead/level_size_before probability Spooner Sec. 3 specifies, and
    confirm zero fills ever occur from a cancel event."""
    rng = np.random.default_rng(0)
    reductions = 0
    trials = 5000
    for _ in range(trials):
        book = FakeBook()
        env = MarketMakingEnv(book, order_size=1000, tick_size=0.005, rng=rng)
        env.reset(mid_price=100.0, book_burn_in=0, warmup_steps=0)
        book.bid_sizes[0] = 60.0
        env.agent_bid = env._new_agent_order('bid', 99.995, book.snapshot())
        pre_snap = book.snapshot()  # level_size_before = 60 (ahead) + implicit agent not counted
        event = {'type': 'cancel', 'side': 'bid', 'level': 0, 'size': 20.0}
        matched_a, matched_b, _, _ = env._resolve_fills(pre_snap, event)
        assert matched_b == 0.0, 'a cancellation must never fill the agent'
        if env.agent_bid['ahead'] < 60.0:
            reductions += 1
    rate = reductions / trials
    expected = 1.0  # ahead(60) / level_size_before(60) = 1.0 -> should ALWAYS reduce here
    assert abs(rate - expected) < 0.02, f'reduction rate {rate} far from expected {expected}'
    print(f'test_cancellation_is_probabilistic_not_a_fill: PASS (reduction rate {rate:.3f})')


def test_inventory_bound_forces_clear():
    from market_qr import QueueReactiveBook, load_calibrated_params
    params = load_calibrated_params()
    book = QueueReactiveBook(params, rng=np.random.default_rng(3))
    env = MarketMakingEnv(book, order_size=1000, min_inv=-2000, max_inv=2000,
                           rng=np.random.default_rng(3))
    env.reset(mid_price=100.0, book_burn_in=1000, warmup_steps=100)
    env.inv = 2500.0  # force out-of-bounds
    state, reward, done, info = env.step(0)  # any action -- should be overridden to MO clear
    assert -2000 <= env.inv <= 2000 or env.inv < 2500, \
        f'inventory {env.inv} was not pulled back toward bounds'
    print(f'test_inventory_bound_forces_clear: PASS (inv {2500.0} -> {env.inv})')


def smoke_test(book_factory, label, n_steps=3000, seed=42):
    rng = np.random.default_rng(seed)
    book = book_factory(rng)
    env = MarketMakingEnv(book, order_size=1000, rng=rng)
    state = env.reset(mid_price=100.0, book_burn_in=2000, warmup_steps=200)

    # A "sticky" random policy (hold an action for ~30 steps before
    # switching) rather than fully random every step -- a fully-random
    # policy re-quotes (and thus resets queue priority) every single step,
    # which makes fills structurally rare regardless of correctness and
    # isn't representative of anything a trained or even a fixed policy
    # would do. This is a much better test of whether fills actually
    # happen under realistic conditions.
    rewards, invs, fills = [], [], 0
    action = rng.integers(0, 9)  # start on a limit-order action, not the MO-clear one
    steps_left = rng.geometric(1 / 30)
    for _ in range(n_steps):
        steps_left -= 1
        if steps_left <= 0:
            action = rng.integers(0, 9)
            steps_left = rng.geometric(1 / 30)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        invs.append(env.inv)
        if info['matched_a'] > 0 or info['matched_b'] > 0:
            fills += 1

    invs = np.array(invs)
    rewards = np.array(rewards)
    print(f'[{label}] {n_steps} steps: mean reward={rewards.mean():.4f} '
          f'std={rewards.std():.4f}  inv range=[{invs.min():.0f}, {invs.max():.0f}]  '
          f'fills={fills} ({100*fills/n_steps:.1f}%)  '
          f'inventory bound respected={bool((invs >= env.min_inv - 1000).all() and (invs <= env.max_inv + 1000).all())}')
    assert not np.isnan(rewards).any(), 'NaN reward encountered'
    assert invs.min() > env.min_inv - 2000 and invs.max() < env.max_inv + 2000, \
        'inventory ran away despite the clearing rule'


def aggressive_fill_test(book_factory, label, n_steps=100000, seed=123):
    """The most aggressive quoting behavior the Table 1 action space
    allows -- action 0 (theta_a=theta_b=1, the tightest permitted
    distance) held constant for the whole run. Unlike smoke_test's sticky
    random policy, this never re-quotes, so queue priority (`ahead`) gets
    a full run's worth of opportunity to decay via cancellations down to
    zero, at which point any market order reaching that price fills the
    agent. This is meant to be decisive, not merely probabilistic: real
    executions are genuinely rare in this calibrated market (~1% of touch
    depletions, confirmed against real IEX data), so smoke_test alone
    showing zero fills over a few thousand steps doesn't distinguish
    "working but rare" from "broken." Under maximum-aggression, sustained
    quoting, zero fills means something is actually broken."""
    rng = np.random.default_rng(seed)
    book = book_factory(rng)
    env = MarketMakingEnv(book, order_size=1000, rng=rng)
    env.reset(mid_price=100.0, book_burn_in=2000, warmup_steps=200)

    fills = 0
    fill_sides = {'bid': 0, 'ask': 0}
    mo_events = 0
    for _ in range(n_steps):
        state, reward, done, info = env.step(0)
        if info['event']['type'] == 'MO':
            mo_events += 1
        if info['matched_a'] > 0:
            fills += 1
            fill_sides['ask'] += 1
        if info['matched_b'] > 0:
            fills += 1
            fill_sides['bid'] += 1

    print(f'[{label}] AGGRESSIVE (action=0 held constant): {n_steps} steps, '
          f'{mo_events} market-order events, {fills} fills '
          f'(bid={fill_sides["bid"]}, ask={fill_sides["ask"]})')
    assert fills > 0, (f'{label}: NO fills occurred under the most aggressive quoting '
                        f'policy over {n_steps} steps and {mo_events} market-order events '
                        f'-- something is broken, this should not be possible if fills work')
    return fills


if __name__ == '__main__':
    test_market_order_partial_fill()
    test_cancellation_is_probabilistic_not_a_fill()
    test_inventory_bound_forces_clear()

    from market_qr import QueueReactiveBook, load_calibrated_params
    from market_zi import ZeroIntelligenceBook, calibrate_zi_params, load_qr_params

    qr_params = load_calibrated_params()
    smoke_test(lambda rng: QueueReactiveBook(qr_params, rng=rng), 'queue-reactive')
    aggressive_fill_test(lambda rng: QueueReactiveBook(qr_params, rng=rng), 'queue-reactive')

    zi_p = calibrate_zi_params(load_qr_params())
    smoke_test(lambda rng: ZeroIntelligenceBook(zi_p['alpha'], zi_p['mu'], zi_p['delta'],
                                                 zi_p['dp'], zi_p['sigma'], n_ticks=200, rng=rng),
               'poisson-control')
    aggressive_fill_test(lambda rng: ZeroIntelligenceBook(zi_p['alpha'], zi_p['mu'], zi_p['delta'],
                                                           zi_p['dp'], zi_p['sigma'], n_ticks=200, rng=rng),
                          'poisson-control')
    print('ALL TESTS PASSED')
