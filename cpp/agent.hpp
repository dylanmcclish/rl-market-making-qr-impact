// The Spooner et al. 2018 consolidated agent -- port of agent_spooner.py.
// Table 2 hyperparameters, verbatim (see that file for the same flagged
// defaults: TILES_PER_DIM=8, replacing traces -- neither specified by the
// paper). The tile coders (agent/market/full) are stateless after
// construction (pure functions of their input, no mutable members), so
// every training thread shares ONE set of them by const reference --
// only the weight vector w_ (shared, unsynchronized -- Hogwild) and each
// thread's own SparseTraces/Rng are per-thread state.
#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "env.hpp"
#include "tile_coding.hpp"
#include "sparse_traces.hpp"
#include "rng.hpp"

constexpr int64_t MEMORY_SIZE = 10'000'000;
constexpr int N_TILINGS = 32;
constexpr int TILES_PER_DIM = 8;  // flagged default, see agent_spooner.py

constexpr double GROUP_WEIGHT_AGENT = 0.6;
constexpr double GROUP_WEIGHT_MARKET = 0.1;
constexpr double GROUP_WEIGHT_FULL = 0.3;

constexpr double ALPHA = 0.001;
constexpr double GAMMA = 0.97;
constexpr double LAMBDA_TRACE = 0.96;
constexpr double EPS_INIT = 0.7;
constexpr double EPS_FLOOR = 0.0001;
constexpr double EPS_T = 1000.0;

constexpr double SPOONER_ORDER_SIZE = 1000.0;
constexpr double SPOONER_MIN_INV = -10000.0;
constexpr double SPOONER_MAX_INV = 10000.0;

constexpr int N_AGENT_VARS = 3;
constexpr int N_MARKET_VARS = 6;
constexpr int N_FULL_VARS = 9;

inline double normalize_clip(double v, double lo, double hi) {
    double x = (v - lo) / (hi - lo);
    return std::min(1.0, std::max(0.0, x));
}

inline void build_agent_vec(const State& s, double* x) {
    x[0] = normalize_clip(s.inventory, SPOONER_MIN_INV, SPOONER_MAX_INV);
    x[1] = normalize_clip(s.theta_a, 1.0, 5.0);
    x[2] = normalize_clip(s.theta_b, 1.0, 5.0);
}
inline void build_market_vec(const State& s, double* x) {
    x[0] = normalize_clip(s.spread, 0.0, 0.10);
    x[1] = normalize_clip(s.mid_price_move, -0.02, 0.02);
    x[2] = normalize_clip(s.imbalance, -1.0, 1.0);
    x[3] = normalize_clip(s.signed_volume, -1000.0, 1000.0);
    x[4] = normalize_clip(s.volatility, 0.0, 0.01);
    x[5] = normalize_clip(s.rsi, 0.0, 100.0);
}
inline void build_full_vec(const State& s, double* x) {
    x[0] = normalize_clip(s.inventory, SPOONER_MIN_INV, SPOONER_MAX_INV);
    x[1] = normalize_clip(s.theta_a, 1.0, 5.0);
    x[2] = normalize_clip(s.theta_b, 1.0, 5.0);
    x[3] = normalize_clip(s.spread, 0.0, 0.10);
    x[4] = normalize_clip(s.mid_price_move, -0.02, 0.02);
    x[5] = normalize_clip(s.imbalance, -1.0, 1.0);
    x[6] = normalize_clip(s.signed_volume, -1000.0, 1000.0);
    x[7] = normalize_clip(s.volatility, 0.0, 0.01);
    x[8] = normalize_clip(s.rsi, 0.0, 100.0);
}

// Holds all-action tile indices for one act()/q_all_actions() call --
// caller-owned scratch space, reused across steps to avoid reallocating
// every call.
struct ActionIndices {
    int64_t agent[N_TILINGS * N_ACTIONS];
    int64_t market[N_TILINGS * N_ACTIONS];
    int64_t full[N_TILINGS * N_ACTIONS];
};

class SpoonerAgent {
public:
    // w: shared weight buffer (size memory_size), NOT owned -- for Hogwild
    // training this is the one buffer every worker thread writes into
    // with no locking, matching main.cpp's train() in the real repo.
    //
    // basic_state: when true, this is Spooner's "basic agent" state
    // representation -- a single tiling over agent-state only (inventory,
    // theta_a, theta_b), full weight 1.0, no market/full groups computed
    // at all (real compute savings, not just a zeroed-out weight) -- vs.
    // the default LCTC (agent+market+full, weights 0.6/0.1/0.3) the
    // consolidated agent uses. The learning ALGORITHM (SARSA(lambda) with
    // replacing traces) is unchanged either way, per this project's own
    // scoping: only state encoding and reward function vary across the
    // basic/consolidated comparison, not the algorithm itself.
    SpoonerAgent(float* w, int64_t memory_size,
                 const HashingTileCoder& agent_coder, const HashingTileCoder& market_coder,
                 const HashingTileCoder& full_coder, uint64_t rng_seed, bool basic_state = false)
        : w_(w), memory_size_(memory_size),
          agent_coder_(agent_coder), market_coder_(market_coder), full_coder_(full_coder),
          traces_((int32_t) memory_size), rng_(rng_seed), basic_state_(basic_state) {}

    void q_all_actions(const State& s, double* qs_out, ActionIndices& idx) const {
        double xa[N_AGENT_VARS];
        build_agent_vec(s, xa);
        agent_coder_.active_indices_all_actions(xa, N_ACTIONS, idx.agent);

        if (basic_state_) {
            for (int a = 0; a < N_ACTIONS; a++) qs_out[a] = 0.0;
            for (int t = 0; t < N_TILINGS; t++)
                for (int a = 0; a < N_ACTIONS; a++)
                    qs_out[a] += w_[idx.agent[t * N_ACTIONS + a]];
            return;
        }

        double xm[N_MARKET_VARS], xf[N_FULL_VARS];
        build_market_vec(s, xm);
        build_full_vec(s, xf);
        market_coder_.active_indices_all_actions(xm, N_ACTIONS, idx.market);
        full_coder_.active_indices_all_actions(xf, N_ACTIONS, idx.full);

        for (int a = 0; a < N_ACTIONS; a++) qs_out[a] = 0.0;
        for (int t = 0; t < N_TILINGS; t++) {
            for (int a = 0; a < N_ACTIONS; a++) {
                int off = t * N_ACTIONS + a;
                qs_out[a] += GROUP_WEIGHT_AGENT * w_[idx.agent[off]]
                           + GROUP_WEIGHT_MARKET * w_[idx.market[off]]
                           + GROUP_WEIGHT_FULL * w_[idx.full[off]];
            }
        }
    }

    // eps_t defaults to Table 2's 1000, but is a real runtime parameter --
    // used to rescale the exploration schedule when training runs shorter
    // than Spooner's own 1000-episode protocol (eps_t scaled by the same
    // factor as the episode count reproduces an identical epsilon
    // trajectory, just compressed -- see the derivation in the training
    // driver script's comments).
    static double epsilon(double episode, double eps_t = EPS_T) {
        return EPS_FLOOR + (EPS_INIT - EPS_FLOOR) * eps_t / (eps_t + episode);
    }

    // Epsilon-greedy over the qs already computed by q_all_actions.
    int act_from_qs(const double* qs, double episode, double eps_t = EPS_T) {
        double eps = epsilon(episode, eps_t);
        if (rng_.uniform() < eps) return rng_.uniform_int(0, N_ACTIONS);
        int best = 0; double bestq = qs[0];
        for (int a = 1; a < N_ACTIONS; a++) if (qs[a] > bestq) { bestq = qs[a]; best = a; }
        return best;
    }

    // One semi-gradient SARSA(lambda) step. `idx`/`action` identify the
    // (state, action_taken) tile set for the step just executed -- action
    // is sliced out of the already-computed all-actions arrays (no need
    // to recompute tile indices even when the environment overrode the
    // agent's proposed action, since q_all_actions already covers every
    // action including MO_CLEAR_ACTION).
    double update(const ActionIndices& idx, int action, double reward, double next_q, bool done) {
        int64_t agent_idx[N_TILINGS];
        for (int t = 0; t < N_TILINGS; t++) agent_idx[t] = idx.agent[t * N_ACTIONS + action];

        if (basic_state_) {
            double q_sa = 0.0;
            for (int t = 0; t < N_TILINGS; t++) q_sa += w_[agent_idx[t]];
            double delta = reward + (done ? 0.0 : GAMMA * next_q) - q_sa;
            traces_.set(agent_idx, N_TILINGS, 1.0f);
            if (delta != 0.0) traces_.apply_update(w_, (float) (ALPHA * delta));
            traces_.decay_and_prune((float) (GAMMA * LAMBDA_TRACE));
            if (done) traces_.clear();
            return delta;
        }

        int64_t market_idx[N_TILINGS], full_idx[N_TILINGS];
        for (int t = 0; t < N_TILINGS; t++) {
            market_idx[t] = idx.market[t * N_ACTIONS + action];
            full_idx[t] = idx.full[t * N_ACTIONS + action];
        }
        double q_sa = 0.0;
        for (int t = 0; t < N_TILINGS; t++)
            q_sa += GROUP_WEIGHT_AGENT * w_[agent_idx[t]]
                  + GROUP_WEIGHT_MARKET * w_[market_idx[t]]
                  + GROUP_WEIGHT_FULL * w_[full_idx[t]];

        double delta = reward + (done ? 0.0 : GAMMA * next_q) - q_sa;

        traces_.set(agent_idx, N_TILINGS, (float) GROUP_WEIGHT_AGENT);
        traces_.set(market_idx, N_TILINGS, (float) GROUP_WEIGHT_MARKET);
        traces_.set(full_idx, N_TILINGS, (float) GROUP_WEIGHT_FULL);

        if (delta != 0.0) traces_.apply_update(w_, (float) (ALPHA * delta));
        traces_.decay_and_prune((float) (GAMMA * LAMBDA_TRACE));
        if (done) traces_.clear();
        return delta;
    }

    void clear_traces() { traces_.clear(); }
    Rng& rng() { return rng_; }

private:
    float* w_;
    int64_t memory_size_;
    const HashingTileCoder &agent_coder_, &market_coder_, &full_coder_;
    SparseTraces traces_;
    Rng rng_;
    bool basic_state_;
};
