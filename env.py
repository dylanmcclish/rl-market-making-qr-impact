"""
Environment wrapper (Part 3): a thin, event-driven Gym-style
(reset()/step(action)) environment implementing the mechanics Spooner et
al. Section 4 describes, on top of either simulator from Parts 1-2 --
market_qr.QueueReactiveBook or market_zi.ZeroIntelligenceBook, both
duck-typed with reset()/step_background()/snapshot().

Design decisions worth being explicit about (the paper describes these in
prose, not pseudocode):

- One RL step = one background LOB event. The agent "acts on events as
  they occur ... actions are not spaced regularly in time" (Sec. 4.1) --
  so the step boundary IS the background event boundary, not a fixed
  clock tick.

- The agent's own resting orders never touch the background simulator's
  arrays. Spooner Sec. 3 states explicitly that the agent's orders "cannot
  impact the market" since their size is small relative to real volume --
  so the agent's queue position is tracked as an OVERLAY: an `ahead`
  count (how much real volume sits in front of the agent's order at its
  price) plus the agent's own remaining size, updated against each
  background event using the "cancellations distributed uniformly" rule
  Sec. 3 states explicitly:
    - a cancellation at the agent's price is a coin flip weighted by
      ahead / level_size_before -- heads, it came from in front of the
      agent (ahead -= removed); tails, it came from behind (no effect).
    - a market order consumes FIFO from the front, so it's deterministic:
      it eats into `ahead` first, then the agent's own size (a fill),
      then whatever's behind -- no randomness needed there.
    - a new limit order arrival always joins the back of the queue --
      never affects the agent's position.
  Every removal event is resolved this way against a fresh recomputation
  of the FIFO walk from the *pre-event* book snapshot (not by trusting
  the simulator's post-mutation internal level indices, which can shift
  under a level collapse mid-event).

- Inventory constraint (Table 2: [-10000, 10000]): once breached, the
  paper says "trading is restricted to orders that bring the agent closer
  to a neutral position" -- implemented as forcing action 9 (the
  inventory-clearing market order, which is exactly that by definition)
  whenever inventory is outside the bound, overriding whatever the policy
  chose.

- Rolling-window lengths for the market-state features (Spread's own
  moving average, mid-price-move/volatility/signed-volume windows) aren't
  in the paper's Table 2 -- only the repo's example config gives any of
  these, and only for 3 of them. Flagged defaults below, matching that
  config where it has a value and choosing a standard default (RSI's
  usual 14-period) where it doesn't.
"""
import numpy as np


ACTION_THETA = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),  # 0-4: symmetric (theta_ask, theta_bid)
    (1, 3), (3, 1), (2, 5), (5, 2),          # 5-8: skewed
]
N_ACTIONS = 10
MO_CLEAR_ACTION = 9


class RollingWindow:
    """Fixed-length rolling buffer with O(1) mean/std -- used for the
    market-state features that need a moving average or a windowed
    volatility/signed-volume estimate."""
    def __init__(self, length):
        self.length = length
        self.buf = []

    def push(self, x):
        self.buf.append(x)
        if len(self.buf) > self.length:
            self.buf.pop(0)

    def mean(self):
        return float(np.mean(self.buf)) if self.buf else 0.0

    def std(self):
        return float(np.std(self.buf)) if len(self.buf) > 1 else 0.0


class MarketMakingEnv:
    def __init__(self, book, order_size=1000, tick_size=0.005,
                 spread_lookback=50, mpm_lookback=15, vol_lookback=60,
                 svl_lookback=60, rsi_period=14,
                 min_inv=-10000, max_inv=10000, mo_clear_alpha=1.0,
                 reward='asym_damped', damping_eta=0.6, rng=None):
        self.book = book
        self.order_size = order_size
        self.tick_size = tick_size
        self.min_inv = min_inv
        self.max_inv = max_inv
        self.mo_clear_alpha = mo_clear_alpha
        self.reward_mode = reward
        self.eta = damping_eta
        self.rng = rng if rng is not None else np.random.default_rng()

        self._half_spread_win = RollingWindow(spread_lookback)
        self._mpm_win = RollingWindow(mpm_lookback)
        self._ret_win = RollingWindow(vol_lookback)
        self._svl_win = RollingWindow(svl_lookback)
        self._rsi_period = rsi_period
        self._rsi_gains = RollingWindow(rsi_period)
        self._rsi_losses = RollingWindow(rsi_period)

        self.inv = 0.0
        self.cash = 0.0
        self.prev_mid = None
        self.agent_bid = None  # {'price', 'ahead', 'remaining'}
        self.agent_ask = None
        self.theta_a = 1
        self.theta_b = 1

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def reset(self, mid_price=100.0, book_burn_in=3000, warmup_steps=200):
        snap = self.book.reset(mid_price=mid_price, burn_in_events=book_burn_in)
        self.inv = 0.0
        self.cash = 0.0
        self.prev_mid = snap['mid']
        self.agent_bid = None
        self.agent_ask = None
        self.theta_a = 1
        self.theta_b = 1

        for w in (self._half_spread_win, self._mpm_win, self._ret_win, self._svl_win,
                  self._rsi_gains, self._rsi_losses):
            w.buf.clear()
        self._half_spread_win.push(snap['spread'] / 2.0)

        # warm up the rolling windows with pure background events before
        # the agent starts quoting, so the first real steps see sane
        # market-state features rather than a cold start.
        for _ in range(warmup_steps):
            dt, event = self.book.step_background()
            self._update_market_state(dt, event)

        return self._get_state()

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def step(self, action):
        requested_action = action
        if self.inv >= self.max_inv or self.inv <= self.min_inv:
            action = MO_CLEAR_ACTION

        if action == MO_CLEAR_ACTION:
            self._cancel_agent_orders()
            fill_qty, avg_price = self._execute_agent_market_order()
            reward_fill_a, reward_fill_b = 0.0, 0.0
            if fill_qty != 0:
                if fill_qty < 0:  # sold to reduce a long position
                    self.inv += fill_qty
                    self.cash -= fill_qty * avg_price  # fill_qty<0 -> cash increases
                else:  # bought to cover a short position
                    self.inv += fill_qty
                    self.cash -= fill_qty * avg_price
        else:
            theta_a, theta_b = ACTION_THETA[action]
            self.theta_a, self.theta_b = theta_a, theta_b
            self._place_agent_orders(theta_a, theta_b)

        pre_snap = self.book.snapshot()
        dt, event = self.book.step_background()
        matched_a, matched_b, p_a, p_b = self._resolve_fills(pre_snap, event)
        self._update_market_state(dt, event)

        mid = self.book.snapshot()['mid']
        delta_m = mid - self.prev_mid
        self.prev_mid = mid

        psi_a = matched_a * (p_a - mid) if matched_a else 0.0
        psi_b = matched_b * (mid - p_b) if matched_b else 0.0
        psi = psi_a + psi_b + self.inv * delta_m

        if self.reward_mode == 'pnl':
            reward = psi
        elif self.reward_mode == 'sym_damped':
            reward = psi - self.eta * self.inv * delta_m
        else:  # asym_damped (the consolidated agent's reward)
            reward = psi - max(0.0, self.eta * self.inv * delta_m)

        state = self._get_state()
        info = {'event': event, 'dt': dt, 'matched_a': matched_a, 'matched_b': matched_b,
                'psi': psi, 'mid': mid, 'action_taken': action, 'action_requested': requested_action}
        done = False  # episode length/termination is a Part 5 training-loop concern
        return state, reward, done, info

    # ------------------------------------------------------------------ #
    # order placement
    # ------------------------------------------------------------------ #
    def _round_to_tick(self, price):
        return round(price / self.tick_size) * self.tick_size

    def _place_agent_orders(self, theta_a, theta_b):
        """Only cancels+replaces a side when its target price actually
        moves (or no order is resting there yet). An unchanged price keeps
        the existing order -- including its accumulated queue priority
        (`ahead`) and whatever size a partial fill left -- exactly as a
        real exchange would: nothing about re-choosing the same action
        touches an order already resting at that price."""
        snap = self.book.snapshot()
        ref = snap['mid']
        # Spread(t_i) = moving average of the market HALF-spread s(t_i)/2
        # (Sec. 4.1, explicit) -- NOT the full spread.
        spread_scale = self._half_spread_win.mean()
        if spread_scale <= 0:
            spread_scale = snap['spread'] / 2.0

        ask_price = self._round_to_tick(ref + theta_a * spread_scale)
        bid_price = self._round_to_tick(ref - theta_b * spread_scale)

        ask_price = max(ask_price, snap['bid_prices'][0] + self.tick_size)
        bid_price = min(bid_price, snap['ask_prices'][0] - self.tick_size)

        if self.agent_ask is None or not np.isclose(self.agent_ask['price'], ask_price,
                                                      atol=self.tick_size / 2):
            self.agent_ask = self._new_agent_order('ask', ask_price, snap)
        if self.agent_bid is None or not np.isclose(self.agent_bid['price'], bid_price,
                                                      atol=self.tick_size / 2):
            self.agent_bid = self._new_agent_order('bid', bid_price, snap)

    def _new_agent_order(self, side, price, snap):
        prices = snap['ask_prices'] if side == 'ask' else snap['bid_prices']
        sizes = snap['ask_sizes'] if side == 'ask' else snap['bid_sizes']
        idx = np.where(np.isclose(prices, price, atol=self.tick_size / 2))[0]
        ahead = float(sizes[idx[0]]) if len(idx) else 0.0
        return {'price': price, 'ahead': ahead, 'remaining': float(self.order_size)}

    def _cancel_agent_orders(self):
        self.agent_bid = None
        self.agent_ask = None

    def _execute_agent_market_order(self):
        """Action 9: clear inventory with a market order sized
        Size_m = -alpha * Inv(t) (Sec. 4.1), walking the real book's
        current depth for the fill price -- doesn't touch the background
        simulator's own book, matching the no-market-impact assumption."""
        if self.inv == 0:
            return 0.0, 0.0
        size = -self.mo_clear_alpha * self.inv
        snap = self.book.snapshot()
        side_prices = snap['ask_prices'] if size > 0 else snap['bid_prices']
        side_sizes = snap['ask_sizes'] if size > 0 else snap['bid_sizes']
        remaining = abs(size)
        cost = 0.0
        filled = 0.0
        for p, s in zip(side_prices, side_sizes):
            take = min(remaining, s)
            if take <= 0:
                continue
            cost += take * p
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled == 0:
            return 0.0, 0.0
        avg_price = cost / filled
        signed_fill = filled if size > 0 else -filled
        return signed_fill, avg_price

    # ------------------------------------------------------------------ #
    # fill resolution against background events
    # ------------------------------------------------------------------ #
    def _resolve_fills(self, pre_snap, event):
        matched_a = matched_b = 0.0
        p_a = self.agent_ask['price'] if self.agent_ask else None
        p_b = self.agent_bid['price'] if self.agent_bid else None

        if event['type'] == 'MO':
            # Deterministic FIFO walk, done inline (not via _decompose_event)
            # because a market order that reaches the agent's price must
            # pause there and consume ahead -> agent's own resting size (a
            # real fill) -> whatever real volume sits behind the agent --
            # all at that ONE price -- before ever moving to the next tick.
            # The agent's own resting size is invisible to the book's
            # arrays (Sec. 3: agent orders can't impact the market), so
            # `sizes[level]` alone conflates "ahead" and "behind"; only the
            # env's own tracked `ahead` value can split them apart.
            consumed_side = 'ask' if event['side'] == 'buy' else 'bid'
            agent_order = self.agent_ask if consumed_side == 'ask' else self.agent_bid
            prices = pre_snap['ask_prices'] if consumed_side == 'ask' else pre_snap['bid_prices']
            sizes = pre_snap['ask_sizes'] if consumed_side == 'ask' else pre_snap['bid_sizes']
            remaining = event['size']
            for p, s in zip(prices, sizes):
                if remaining <= 0:
                    break
                is_agent_level = (agent_order is not None
                                   and np.isclose(p, agent_order['price'], atol=self.tick_size / 2))
                if is_agent_level:
                    ahead_before = agent_order['ahead']
                    behind_before = max(0.0, s - ahead_before)
                    eat_ahead = min(ahead_before, remaining)
                    agent_order['ahead'] = ahead_before - eat_ahead
                    remaining -= eat_ahead
                    fill = min(agent_order['remaining'], remaining)
                    agent_order['remaining'] -= fill
                    remaining -= fill
                    if fill > 0:
                        if consumed_side == 'ask':
                            matched_a += fill
                        else:
                            matched_b += fill
                    remaining -= min(behind_before, remaining)
                    if agent_order['remaining'] <= 0:
                        if consumed_side == 'ask':
                            self.agent_ask = None
                        else:
                            self.agent_bid = None
                        agent_order = None
                else:
                    remaining -= min(s, remaining)

            if matched_a > 0:
                self.inv -= matched_a
                self.cash += matched_a * p_a
            if matched_b > 0:
                self.inv += matched_b
                self.cash -= matched_b * p_b

            return matched_a, matched_b, p_a, p_b

        for side, price, qty, level_size_before, is_removal in self._decompose_event(pre_snap, event):
            if is_removal and side == 'ask' and self.agent_ask is not None \
                    and np.isclose(price, self.agent_ask['price'], atol=self.tick_size / 2):
                self._apply_cancel_to_agent_order(self.agent_ask, qty, level_size_before)
            if is_removal and side == 'bid' and self.agent_bid is not None \
                    and np.isclose(price, self.agent_bid['price'], atol=self.tick_size / 2):
                self._apply_cancel_to_agent_order(self.agent_bid, qty, level_size_before)

        return matched_a, matched_b, p_a, p_b

    def _apply_cancel_to_agent_order(self, order, qty, level_size_before):
        """A cancellation never fills the agent -- we don't know which
        specific resting order was pulled, so per Spooner Sec. 3, treat it
        as having come from ahead of the agent with probability
        ahead/level_size_before, otherwise leave `ahead` unchanged."""
        if level_size_before <= 0:
            return
        p_ahead = min(1.0, order['ahead'] / level_size_before)
        if self.rng.random() < p_ahead:
            order['ahead'] = max(0.0, order['ahead'] - qty)

    def _decompose_event(self, pre_snap, event):
        """Re-derive a (side, price, qty, level_size_before, is_removal)
        tuple for a single-level LO/cancel event, purely from the
        PRE-event snapshot rather than trusting the simulator's own
        post-mutation level index. MO events are handled separately,
        inline in _resolve_fills, since they need a multi-level walk that
        can pause at the agent's own price (see that method's docstring)."""
        out = []
        t = event['type']
        if t in ('LO', 'cancel'):
            side = event['side']
            prices = pre_snap['ask_prices'] if side == 'ask' else pre_snap['bid_prices']
            sizes = pre_snap['ask_sizes'] if side == 'ask' else pre_snap['bid_sizes']
            level = event['level']
            if level < len(prices):
                price = prices[level]
                size_before = sizes[level]
                out.append((side, price, event.get('size', 0.0), size_before, t == 'cancel'))
        return out

    # ------------------------------------------------------------------ #
    # market-state features
    # ------------------------------------------------------------------ #
    def _update_market_state(self, dt, event):
        snap = self.book.snapshot()
        self._half_spread_win.push(snap['spread'] / 2.0)
        mid = snap['mid']
        d_mid = mid - self.prev_mid if self.prev_mid is not None else 0.0
        self._mpm_win.push(d_mid)
        self._ret_win.push(d_mid)
        signed_vol = 0.0
        if event['type'] == 'MO':
            signed_vol = event['size'] if event['side'] == 'buy' else -event['size']
        self._svl_win.push(signed_vol)
        self._rsi_gains.push(max(0.0, d_mid))
        self._rsi_losses.push(max(0.0, -d_mid))

    def _rsi(self):
        avg_gain = self._rsi_gains.mean()
        avg_loss = self._rsi_losses.mean()
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _imbalance(self, snap):
        bid_depth = snap['bid_sizes'].sum()
        ask_depth = snap['ask_sizes'].sum()
        total = bid_depth + ask_depth
        return (bid_depth - ask_depth) / total if total > 0 else 0.0

    def _get_state(self):
        snap = self.book.snapshot()
        return {
            # agent-state (3)
            'inventory': self.inv,
            'theta_a': self.theta_a,
            'theta_b': self.theta_b,
            # market-state (6) -- the full set the paper text lists (Sec 4.3)
            'spread': snap['spread'],
            'mid_price_move': self._mpm_win.mean(),
            'imbalance': self._imbalance(snap),
            'signed_volume': self._svl_win.mean(),
            'volatility': self._ret_win.std(),
            'rsi': self._rsi(),
        }
