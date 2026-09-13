"""
Generic hashed tile coding (CMAC), used by agent_spooner.py to build the
three independent tilings the LCTC state representation needs (Sec. 4.3:
agent-state, market-state, full-state).

Standard tile coding: each of M independent tilings partitions the
(normalized) state space into a uniform grid, offset from the others by a
different fractional displacement so their tile boundaries don't align.
An observation activates exactly one tile per tiling -- M active tiles per
state overall. The paper's memory_size=10^7 is far smaller than the true
number of distinct tiles across all (group, tiling, action) combinations
at any reasonable resolution, so tile identities are hashed down into that
fixed-size table (accepting occasional hash collisions) rather than
mapped one-to-one -- this is the standard technique Sutton & Barto's own
reference tile-coding software uses, not a shortcut specific to this repo.
"""
import numpy as np


class HashingTileCoder:
    def __init__(self, n_dims, n_tilings, tiles_per_dim, memory_size, group_id, seed=0):
        self.n_dims = n_dims
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.memory_size = memory_size
        self.group_id = group_id
        # Per Sutton & Barto's displacement scheme: tiling i is offset by
        # i * (2j+1) / n_tilings along dimension j, using the first n_dims
        # odd numbers -- guarantees offsets are well spread rather than
        # colliding for small n_tilings.
        odds = 2 * np.arange(n_dims) + 1
        self.offsets = (np.arange(n_tilings)[:, None] * odds[None, :]) % n_tilings
        self.offsets = self.offsets / n_tilings  # in [0, 1)
        self._rng = np.random.default_rng(seed)
        # A random salt per (tiling) so different HashingTileCoder
        # instances (agent/market/full groups) don't alias into the same
        # region of the shared memory table even with identical inputs.
        self._salt = self._rng.integers(0, 2**31 - 1, size=n_tilings)

        # ---- precomputed constants for the vectorized hash (see
        # active_indices_all_actions) -- the same polynomial hash as
        # active_indices below, computed once at construction time instead
        # of on every call. h = (((salt*mult + group_id)*mult + action)*mult
        # + c_0)*mult + c_1 ... expands, under mod-2**64 arithmetic, to a
        # sum of each term times a fixed power of mult -- so the whole
        # per-tiling, per-action hash becomes a couple of vectorized numpy
        # ops instead of a Python loop over tilings with an inner loop over
        # dims, called once per action.
        mult = np.uint64(1000003)
        with np.errstate(over='ignore'):
            powers = np.empty(n_dims + 3, dtype=np.uint64)
            powers[0] = np.uint64(1)
            for i in range(1, n_dims + 3):
                powers[i] = powers[i - 1] * mult
            self._mult = mult
            self._dim_pow = powers[0:n_dims][::-1].copy()  # dim i -> mult**(n_dims-1-i)
            self._action_pow = powers[n_dims]
            self._base_const = (np.uint64(self._salt) * powers[n_dims + 2]
                                 + np.uint64(group_id) * powers[n_dims + 1])

    def active_indices_all_actions(self, x_norm, n_actions):
        """Vectorized equivalent of calling active_indices(x_norm, a) for
        every a in range(n_actions): returns an (n_tilings, n_actions)
        array of hashed indices. Mathematically identical to the per-action
        loop (mod-2**64 polynomial hashing is exactly associative/
        distributive), verified against it in test_tile_coding.py -- just
        computed without any Python-level loop over tilings or actions."""
        with np.errstate(over='ignore'):
            coords = np.floor((x_norm[None, :] + self.offsets) * self.tiles_per_dim).astype(np.int64)
            coords = np.clip(coords, 0, self.tiles_per_dim - 1).astype(np.uint64)
            weighted = coords * self._dim_pow[None, :]
            base = self._base_const + weighted.sum(axis=1)  # (n_tilings,)
            action_terms = np.arange(n_actions, dtype=np.uint64) * self._action_pow
            h = base[:, None] + action_terms[None, :]  # (n_tilings, n_actions)
            idx = (h % np.uint64(self.memory_size)).astype(np.int64)
        return idx

    def active_indices(self, x_norm, action):
        """x_norm: array of n_dims values already normalized to [0, 1]
        (clipped by the caller). Returns n_tilings hashed indices into
        [0, memory_size) -- the active tile in each tiling, for this
        (state, action) pair. Hashing uses fixed-width uint64 arithmetic
        (wraps on overflow, like a real hash function) rather than Python
        big-ints, which would otherwise grow unboundedly and get slow."""
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
        mult = np.uint64(1000003)
        idx = np.empty(self.n_tilings, dtype=np.int64)
        with np.errstate(over='ignore'):  # wraparound is the intended hash behavior
            for t in range(self.n_tilings):
                coords = np.floor((x_norm + self.offsets[t]) * self.tiles_per_dim).astype(np.int64)
                coords = np.clip(coords, 0, self.tiles_per_dim - 1)
                h = np.uint64(self._salt[t])
                h = (h * mult + np.uint64(self.group_id)) & mask
                h = (h * mult + np.uint64(action)) & mask
                for c in coords:
                    h = (h * mult + np.uint64(int(c))) & mask
                idx[t] = int(h % np.uint64(self.memory_size))
        return idx
