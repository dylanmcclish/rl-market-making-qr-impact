"""
Bounded, array-based sparse eligibility traces -- replaces agent_spooner.py's
original Python dict of traces with the same structure the actual
tspooner/rl_markets C++ agent uses (src/rl/traces.cpp): a flat eligibility
array plus a compact list of currently-nonzero feature indices (with an
O(1) inverse map for removal), so decay/prune/apply-update are all single
vectorized numpy operations over just the active entries, regardless of how
sparse or dense memory_size is. A dict does the same job logically but pays
per-key Python/hashing overhead on every step; this doesn't.

Semantics are unchanged from the dict version: `set(idx, value)` is a
*replacing* trace (overwrites, doesn't accumulate), and entries below
`tolerance` after decay are dropped. `max_nonzero` is a safety bound (with
headroom well above the ~25k steady-state active count implied by
gamma*lambda's ~250-step decay horizon here) -- if it's ever hit, tolerance
is raised to prune harder, mirroring collision_table's own adaptive
tolerance in the C++ traces implementation, rather than growing unboundedly.
"""
import numpy as np


class SparseTraces:
    def __init__(self, memory_size, max_nonzero=200_000, tolerance=1e-8):
        self.memory_size = memory_size
        self.max_nonzero = max_nonzero
        self.tolerance = tolerance

        self.eligibility = np.zeros(memory_size, dtype=np.float32)
        self.nonzero_idx = np.zeros(max_nonzero, dtype=np.int64)
        self.pos_in_list = np.full(memory_size, -1, dtype=np.int64)
        self.n_active = 0

    def set(self, idx_array, value):
        """Replacing trace: every feature in idx_array gets eligibility =
        value (not +=), and is registered as active if it wasn't already."""
        for i in idx_array:
            i = int(i)
            pos = self.pos_in_list[i]
            if pos == -1:
                if self.n_active >= self.max_nonzero:
                    self._increase_tolerance()
                self.pos_in_list[i] = self.n_active
                self.nonzero_idx[self.n_active] = i
                self.n_active += 1
            self.eligibility[i] = value

    def apply_update(self, w, scaled_delta):
        """w[i] += scaled_delta * eligibility[i] for every active i --
        one vectorized fancy-index op over the active set."""
        if self.n_active == 0:
            return
        active = self.nonzero_idx[:self.n_active]
        w[active] += scaled_delta * self.eligibility[active]

    def decay_and_prune(self, rate):
        """eligibility *= rate for every active entry, then drop anything
        that fell below tolerance -- via a boolean-mask compaction rather
        than per-element removal, so cost is O(n_active) numpy ops, not
        O(n_active) Python-level ones."""
        if self.n_active == 0:
            return
        active = self.nonzero_idx[:self.n_active]
        self.eligibility[active] *= rate
        keep = self.eligibility[active] >= self.tolerance
        if keep.all():
            return
        removed = active[~keep]
        self.eligibility[removed] = 0.0
        self.pos_in_list[removed] = -1
        kept = active[keep]
        n_kept = len(kept)
        self.nonzero_idx[:n_kept] = kept
        self.pos_in_list[kept] = np.arange(n_kept)
        self.n_active = n_kept

    def clear(self):
        if self.n_active == 0:
            return
        active = self.nonzero_idx[:self.n_active]
        self.eligibility[active] = 0.0
        self.pos_in_list[active] = -1
        self.n_active = 0

    def _increase_tolerance(self):
        self.tolerance *= 1.1
        active = self.nonzero_idx[:self.n_active]
        keep = self.eligibility[active] >= self.tolerance
        if keep.all():
            return
        removed = active[~keep]
        self.eligibility[removed] = 0.0
        self.pos_in_list[removed] = -1
        kept = active[keep]
        n_kept = len(kept)
        self.nonzero_idx[:n_kept] = kept
        self.pos_in_list[kept] = np.arange(n_kept)
        self.n_active = n_kept
