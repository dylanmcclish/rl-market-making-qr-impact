"""
Part 4: the Spooner et al. 2018 "consolidated agent," unchanged --
SARSA(lambda) over a linear combination of tile codings (LCTC, Sec. 4.3),
Table 1's action space, and the asymmetrically dampened PnL reward
(already implemented in env.py's reward_mode='asym_damped').

Every numeric hyperparameter below is Table 2's, verbatim -- see
STATE_RANGES and the two flagged defaults (trace type, epsilon schedule
shape) for the handful of details Table 2 and the paper's prose don't
specify; both are called out explicitly rather than silently guessed.
"""
import numpy as np

from env import N_ACTIONS, MO_CLEAR_ACTION
from tile_coding import HashingTileCoder
from sparse_traces import SparseTraces

# ---- Table 2, verbatim ------------------------------------------------
MEMORY_SIZE = 10_000_000
N_TILINGS = 32
GROUP_WEIGHTS = {'agent': 0.6, 'market': 0.1, 'full': 0.3}
ALPHA = 0.001
GAMMA = 0.97
LAMBDA_TRACE = 0.96
EPS_INIT = 0.7
EPS_FLOOR = 0.0001
EPS_T = 1000
ORDER_SIZE = 1000
MIN_INV, MAX_INV = -10000, 10000

# ---- flagged defaults (not specified in the paper) ---------------------
TILES_PER_DIM = 8  # tile-coding resolution per state dimension; Table 2 gives
                    # tiling COUNT (32) and total memory (10^7) but not this.
TRACE_TYPE = 'replacing'  # standard choice for tile-coding SARSA(lambda);
                           # the paper doesn't say which trace type it used.

# Normalization ranges for the 9 raw state variables env.py exposes.
# Tile coding needs bounded inputs; these bounds aren't published anywhere
# (not even the repo's example config specifies them) -- chosen from the
# variables' natural ranges (inventory bounds, RSI, imbalance are exact;
# spread/mid-price-move/signed-volume/volatility are set from the real
# INTC data's own observed scale, generously margined, with values beyond
# the range clipped rather than causing an error).
STATE_RANGES = {
    'inventory': (MIN_INV, MAX_INV),
    'theta_a': (1, 5),
    'theta_b': (1, 5),
    'spread': (0.0, 0.10),          # real median ~0.03; ~3x margin
    'mid_price_move': (-0.02, 0.02),
    'imbalance': (-1.0, 1.0),        # exact, by construction
    'signed_volume': (-1000.0, 1000.0),
    'volatility': (0.0, 0.01),
    'rsi': (0.0, 100.0),             # exact, by construction
}

AGENT_VARS = ['inventory', 'theta_a', 'theta_b']
MARKET_VARS = ['spread', 'mid_price_move', 'imbalance', 'signed_volume', 'volatility', 'rsi']
FULL_VARS = AGENT_VARS + MARKET_VARS


def normalize(state, keys):
    x = np.empty(len(keys))
    for i, k in enumerate(keys):
        lo, hi = STATE_RANGES[k]
        v = (state[k] - lo) / (hi - lo)
        x[i] = min(1.0, max(0.0, v))
    return x


class SpoonerAgent:
    def __init__(self, memory_size=MEMORY_SIZE, n_tilings=N_TILINGS,
                 tiles_per_dim=TILES_PER_DIM, seed=0, theta_buffer=None,
                 max_nonzero_traces=200_000):
        """theta_buffer: optional externally-provided buffer (e.g. a
        multiprocessing.RawArray('f', memory_size)) to back the weight
        vector -- so a Hogwild-style training harness (train.py) can hand
        every worker process a numpy view over the SAME shared memory,
        with in-place updates from any process visible to all of them, no
        locking. Defaults to a private zero-initialized array when not
        training in parallel."""
        self.memory_size = memory_size
        if theta_buffer is not None:
            self.w = np.frombuffer(theta_buffer, dtype=np.float32)
            assert self.w.shape[0] == memory_size
        else:
            self.w = np.zeros(memory_size, dtype=np.float32)
        self.traces = SparseTraces(memory_size, max_nonzero=max_nonzero_traces)

        rng_seed = seed
        self.coders = {
            'agent': HashingTileCoder(len(AGENT_VARS), n_tilings, tiles_per_dim,
                                       memory_size, group_id=0, seed=rng_seed),
            'market': HashingTileCoder(len(MARKET_VARS), n_tilings, tiles_per_dim,
                                        memory_size, group_id=1, seed=rng_seed + 1),
            'full': HashingTileCoder(len(FULL_VARS), n_tilings, tiles_per_dim,
                                      memory_size, group_id=2, seed=rng_seed + 2),
        }
        self.rng = np.random.default_rng(seed)

    def _active_all(self, state):
        """Returns {'agent': idx, 'market': idx, 'full': idx}, each an
        (N_TILINGS, N_ACTIONS) array -- every group's tile indices for
        every action at once, via HashingTileCoder's vectorized hash."""
        x_agent = normalize(state, AGENT_VARS)
        x_market = normalize(state, MARKET_VARS)
        x_full = normalize(state, FULL_VARS)
        return {
            'agent': self.coders['agent'].active_indices_all_actions(x_agent, N_ACTIONS),
            'market': self.coders['market'].active_indices_all_actions(x_market, N_ACTIONS),
            'full': self.coders['full'].active_indices_all_actions(x_full, N_ACTIONS),
        }

    def _active_indices(self, state, action):
        """Returns {'agent': idx_array, 'market': idx_array, 'full': idx_array}
        for a single action -- each an array of N_TILINGS hashed indices
        into self.w. Used by update()/tests; act()/q_all_actions use
        _active_all directly since they need every action anyway."""
        all_actions = self._active_all(state)
        return {group: idx[:, action] for group, idx in all_actions.items()}

    def q_value(self, active_by_group):
        """Eq. 7: Q(s,a) = sum_g lambda_g * sum_{i in active tiles of g} w[i]."""
        total = 0.0
        for group, weight in GROUP_WEIGHTS.items():
            idx = active_by_group[group]
            total += weight * self.w[idx].sum()
        return float(total)

    def q_all_actions(self, state):
        """Returns (q_values[N_ACTIONS], active_indices_per_action[N_ACTIONS]).
        Vectorized: computes tile indices and Q for every action in one
        shot per group (fancy-index + sum over the tilings axis), rather
        than looping over actions in Python."""
        all_actions = self._active_all(state)  # group -> (N_TILINGS, N_ACTIONS)
        qs = np.zeros(N_ACTIONS)
        for group, weight in GROUP_WEIGHTS.items():
            idx = all_actions[group]
            qs += weight * self.w[idx].sum(axis=0)
        actives = [{group: idx[:, a] for group, idx in all_actions.items()}
                   for a in range(N_ACTIONS)]
        return qs, actives

    def epsilon(self, episode):
        """Flagged default: hyperbolic decay from EPS_INIT toward EPS_FLOOR
        with EPS_T as the decay time-constant (episode at which epsilon has
        fallen halfway) -- Table 2 gives eps_init/eps_floor/eps_T as
        numbers but the paper's prose doesn't spell out the decay curve's
        functional form."""
        return EPS_FLOOR + (EPS_INIT - EPS_FLOOR) * EPS_T / (EPS_T + episode)

    def act(self, state, episode, qs=None, actives=None):
        """Epsilon-greedy over all N_ACTIONS actions (env.py separately
        enforces the inventory-bound override -- see info['action_taken'];
        the agent's SARSA update must use that, not this proposal, when
        they differ)."""
        if qs is None:
            qs, actives = self.q_all_actions(state)
        eps = self.epsilon(episode)
        if self.rng.random() < eps:
            a = int(self.rng.integers(0, N_ACTIONS))
        else:
            a = int(np.argmax(qs))
        return a, qs, actives

    def update(self, active_by_group, reward, next_q, done):
        """One semi-gradient SARSA(lambda) step with replacing traces.
        `active_by_group` is the (state, action_taken) tile set for the
        step just executed; `next_q` is Q(s', a') for the next state and
        the next action the policy will actually take there (0 if done).
        Trace bookkeeping is delegated to SparseTraces (a bounded,
        array-based sparse structure -- see sparse_traces.py) instead of a
        Python dict; semantics are identical, just vectorized."""
        q_sa = self.q_value(active_by_group)
        delta = reward + (0.0 if done else GAMMA * next_q) - q_sa

        for group, weight in GROUP_WEIGHTS.items():
            self.traces.set(active_by_group[group], weight)  # replacing trace: feature value is lambda_g

        if delta != 0.0:
            self.traces.apply_update(self.w, ALPHA * delta)
        self.traces.decay_and_prune(GAMMA * LAMBDA_TRACE)

        if done:
            self.traces.clear()

        return delta

    def reset_traces(self):
        self.traces.clear()
