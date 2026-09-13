// Diagnostic: what action is the greedy policy actually choosing?
// Runs the greedy (epsilon=0) policy for one episode and tallies an action
// histogram plus inventory-trajectory stats, so a near-zero MAP/PnL result
// from eval.cpp can be distinguished from "learned to sanely avoid the
// touch" vs. "degenerate tie-break landing on a meaningless action."
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "agent.hpp"
#include "env.hpp"
#include "market_qr.hpp"
#include "market_zi.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "tile_coding.hpp"

constexpr uint64_t TILE_SEED_AGENT = 0, TILE_SEED_MARKET = 1, TILE_SEED_FULL = 2;

std::vector<float> load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open weights file: " + path);
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t) f.tellg();
    f.seekg(0);
    std::vector<float> w(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(w.data()), (std::streamsize) bytes);
    return w;
}

const char* action_label(int a) {
    static const char* labels[N_ACTIONS] = {
        "t_ask=1,t_bid=1 (tightest)", "t_ask=2,t_bid=2", "t_ask=3,t_bid=3",
        "t_ask=4,t_bid=4", "t_ask=5,t_bid=5 (widest symmetric)",
        "t_ask=1,t_bid=3 (skew: tight ask/wide bid)",
        "t_ask=3,t_bid=1 (skew: wide ask/tight bid)",
        "t_ask=2,t_bid=5 (skew)", "t_ask=5,t_bid=2 (skew)",
        "MO_CLEAR (inventory-clearing market order)"
    };
    return labels[a];
}

template <typename Book>
void diagnose(Book& book, bool basic_state, const std::vector<float>& weights,
              int64_t n_events, uint64_t seed) {
    HashingTileCoder ac(N_AGENT_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 0, TILE_SEED_AGENT);
    HashingTileCoder mc(N_MARKET_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 1, TILE_SEED_MARKET);
    HashingTileCoder fc(N_FULL_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 2, TILE_SEED_FULL);
    SpoonerAgent agent(const_cast<float*>(weights.data()), (int64_t) weights.size(),
                        ac, mc, fc, seed, basic_state);

    Rng rng(seed);
    MarketMakingEnv<Book> env(book, rng, SPOONER_ORDER_SIZE, 0.005, 50, 15, 60, 60, 14,
                               SPOONER_MIN_INV, SPOONER_MAX_INV);
    State state = env.reset(100.0, 3000, 200);

    int64_t action_count[N_ACTIONS] = {0};
    double inv_sum = 0.0, inv_sq_sum = 0.0, inv_min = 1e18, inv_max = -1e18;
    int64_t n_fills = 0, n_overrides = 0;
    StepInfo info;

    for (int64_t t = 0; t < n_events; t++) {
        double qs[N_ACTIONS];
        ActionIndices idx;
        agent.q_all_actions(state, qs, idx);
        int best = 0; double bestq = qs[0];
        for (int a = 1; a < N_ACTIONS; a++) if (qs[a] > bestq) { bestq = qs[a]; best = a; }

        state = env.step(best, info);
        action_count[info.action_taken]++;
        if (info.action_taken != best) n_overrides++;
        if (info.matched_a > 0.0 || info.matched_b > 0.0) n_fills++;

        double inv = env.inv();
        inv_sum += inv; inv_sq_sum += inv * inv;
        inv_min = std::min(inv_min, inv);
        inv_max = std::max(inv_max, inv);
    }

    double mean_inv = inv_sum / n_events;
    double std_inv = std::sqrt(std::max(0.0, inv_sq_sum / n_events - mean_inv * mean_inv));

    std::printf("  fills=%lld  overrides_to_MO_CLEAR=%lld  inv[min=%.1f mean=%.3f std=%.3f max=%.1f]\n",
                (long long) n_fills, (long long) n_overrides, inv_min, mean_inv, std_inv, inv_max);
    std::printf("  action histogram (%lld total steps):\n", (long long) n_events);
    for (int a = 0; a < N_ACTIONS; a++) {
        double pct = 100.0 * (double) action_count[a] / (double) n_events;
        if (action_count[a] > 0)
            std::printf("    [%d] %-45s %10lld  (%.2f%%)\n", a, action_label(a),
                        (long long) action_count[a], pct);
    }
}

int main(int argc, char** argv) {
    std::string weights_path, book_kind = "qr", state_kind = "full", cfg_dir = ".";
    int64_t events = 5'000'000;
    uint64_t seed = 424242;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--weights") weights_path = next();
        else if (a == "--book") book_kind = next();
        else if (a == "--state") state_kind = next();
        else if (a == "--events") events = std::stoll(next());
        else if (a == "--cfg") cfg_dir = next();
        else if (a == "--seed") seed = std::stoull(next());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    auto weights = load_weights(weights_path);
    bool basic_state = (state_kind == "basic");
    std::printf("=== %s  book=%s state=%s events=%lld ===\n",
                weights_path.c_str(), book_kind.c_str(), state_kind.c_str(), (long long) events);

    if (book_kind == "qr") {
        QRParams qp = load_qr_params(cfg_dir + "/qr_params.txt");
        auto trade_sizes = load_trade_sizes(cfg_dir + "/trade_sizes.txt");
        Rng book_rng(seed + 1);
        QueueReactiveBook book(qp, &trade_sizes, book_rng);
        diagnose(book, basic_state, weights, events, seed);
    } else {
        ZIParams zp = load_zi_params(cfg_dir + "/zi_params.txt");
        Rng book_rng(seed + 1);
        ZeroIntelligenceBook book(zp, book_rng, 200);
        diagnose(book, basic_state, weights, events, seed);
    }
    return 0;
}
