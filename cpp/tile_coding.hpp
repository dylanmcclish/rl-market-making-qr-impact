// Hashed tile coding (CMAC) -- port of tile_coding.py's HashingTileCoder,
// using the same algebraic polynomial-hash form the Python side's
// active_indices_all_actions() uses (verified bit-identical there against
// the naive per-tiling loop): h = salt*mult^(d+2) + group*mult^(d+1) +
// action*mult^d + sum_i coord_i*mult^(d-1-i), mod 2**64. uint64_t overflow
// wraparound is well-defined in C++ (unlike signed overflow), so this is
// safe as written -- no UB, unlike the numpy version which needed
// np.errstate(over='ignore') to suppress a warning about the same,
// perfectly intentional, wraparound.
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class HashingTileCoder {
public:
    HashingTileCoder(int n_dims, int n_tilings, int tiles_per_dim, int64_t memory_size,
                      int group_id, uint64_t seed)
        : n_dims_(n_dims), n_tilings_(n_tilings), tiles_per_dim_(tiles_per_dim),
          memory_size_(memory_size) {
        offsets_.resize((size_t) n_tilings_ * n_dims_);
        for (int t = 0; t < n_tilings_; t++)
            for (int d = 0; d < n_dims_; d++)
                offsets_[(size_t) t * n_dims_ + d] =
                    (double) ((t * (2 * d + 1)) % n_tilings_) / n_tilings_;

        std::mt19937_64 eng(seed);
        std::vector<uint64_t> salt(n_tilings_);
        for (auto& s : salt) s = eng();

        const uint64_t mult = 1000003ULL;
        std::vector<uint64_t> powers(n_dims_ + 3);
        powers[0] = 1ULL;
        for (int i = 1; i < n_dims_ + 3; i++) powers[i] = powers[i - 1] * mult;  // wraps, intentional

        dim_pow_.resize(n_dims_);
        for (int i = 0; i < n_dims_; i++) dim_pow_[i] = powers[n_dims_ - 1 - i];
        action_pow_ = powers[n_dims_];
        uint64_t group_pow = powers[n_dims_ + 1];
        uint64_t salt_pow = powers[n_dims_ + 2];

        base_const_.resize(n_tilings_);
        for (int t = 0; t < n_tilings_; t++)
            base_const_[t] = salt[t] * salt_pow + (uint64_t) group_id * group_pow;
    }

    // Fills idx_out[t * n_actions + a] for t in [0, n_tilings), a in [0, n_actions).
    void active_indices_all_actions(const double* x_norm, int n_actions, int64_t* idx_out) const {
        for (int t = 0; t < n_tilings_; t++) {
            uint64_t h = base_const_[t];
            for (int d = 0; d < n_dims_; d++) {
                double v = (x_norm[d] + offsets_[(size_t) t * n_dims_ + d]) * tiles_per_dim_;
                int c = (int) std::floor(v);
                c = std::min(std::max(c, 0), tiles_per_dim_ - 1);
                h += (uint64_t) c * dim_pow_[d];
            }
            for (int a = 0; a < n_actions; a++) {
                uint64_t hh = h + (uint64_t) a * action_pow_;
                idx_out[(size_t) t * n_actions + a] = (int64_t) (hh % (uint64_t) memory_size_);
            }
        }
    }

private:
    int n_dims_, n_tilings_, tiles_per_dim_;
    int64_t memory_size_;
    std::vector<double> offsets_;
    std::vector<uint64_t> dim_pow_;
    uint64_t action_pow_;
    std::vector<uint64_t> base_const_;
};
