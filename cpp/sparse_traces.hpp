// Bounded array-based sparse eligibility traces -- port of
// sparse_traces.py's SparseTraces (itself modeled on the real
// tspooner/rl_markets C++ agent's src/rl/traces.cpp). int32_t indices
// (not int64_t) since memory_size never exceeds ~2e7, comfortably under
// INT32_MAX, and this halves the per-thread footprint of pos_in_list_
// (the memory_size-length inverse map) -- worth it since every training
// thread owns its own SparseTraces instance.
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

class SparseTraces {
public:
    explicit SparseTraces(int32_t memory_size, int32_t max_nonzero = 200000,
                           float tolerance = 1e-8f)
        : memory_size_(memory_size), max_nonzero_(max_nonzero), tolerance_(tolerance) {
        eligibility_.assign(memory_size_, 0.0f);
        nonzero_idx_.assign(max_nonzero_, 0);
        pos_in_list_.assign(memory_size_, -1);
    }

    void set(const int64_t* idx_array, int count, float value) {
        for (int k = 0; k < count; k++) {
            int32_t i = (int32_t) idx_array[k];
            int32_t pos = pos_in_list_[i];
            if (pos == -1) {
                if (n_active_ >= max_nonzero_) increase_tolerance();
                pos_in_list_[i] = n_active_;
                nonzero_idx_[n_active_] = i;
                n_active_++;
            }
            eligibility_[i] = value;
        }
    }

    void apply_update(float* w, float scaled_delta) {
        for (int32_t k = 0; k < n_active_; k++) {
            int32_t i = nonzero_idx_[k];
            w[i] += scaled_delta * eligibility_[i];
        }
    }

    void decay_and_prune(float rate) {
        int32_t write = 0;
        for (int32_t k = 0; k < n_active_; k++) {
            int32_t i = nonzero_idx_[k];
            eligibility_[i] *= rate;
            if (eligibility_[i] >= tolerance_) {
                nonzero_idx_[write] = i;
                pos_in_list_[i] = write;
                write++;
            } else {
                eligibility_[i] = 0.0f;
                pos_in_list_[i] = -1;
            }
        }
        n_active_ = write;
    }

    void clear() {
        for (int32_t k = 0; k < n_active_; k++) {
            int32_t i = nonzero_idx_[k];
            eligibility_[i] = 0.0f;
            pos_in_list_[i] = -1;
        }
        n_active_ = 0;
    }

    int32_t n_active() const { return n_active_; }

private:
    void increase_tolerance() {
        tolerance_ *= 1.1f;
        int32_t write = 0;
        for (int32_t k = 0; k < n_active_; k++) {
            int32_t i = nonzero_idx_[k];
            if (eligibility_[i] >= tolerance_) {
                nonzero_idx_[write] = i;
                pos_in_list_[i] = write;
                write++;
            } else {
                eligibility_[i] = 0.0f;
                pos_in_list_[i] = -1;
            }
        }
        n_active_ = write;
    }

    int32_t memory_size_, max_nonzero_;
    float tolerance_;
    std::vector<float> eligibility_;
    std::vector<int32_t> nonzero_idx_;
    std::vector<int32_t> pos_in_list_;
    int32_t n_active_ = 0;
};
