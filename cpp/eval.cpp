// Part 5 evaluation -- port of train.py's evaluate_policy()/greedy_policy_fn()/
// fixed_spread_policy_fn()/random_policy_fn(). Loads a trained weight
// vector (the binary float32 file train.cpp's train_hogwild() saves) and
// runs the greedy consolidated agent, plus Spooner's own benchmark
// policies (fixed theta_a=theta_b spread, random), against either
// simulator, reporting the paper's own metrics: normalized daily PnL and
// mean absolute position.
#include <algorithm>
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
constexpr uint64_t EVAL_BASE_SEED = 10000;

struct EvalResult {
    double norm_pnl_mean, norm_pnl_std, map_mean;
};

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

template <typename Book, typename BookFactory, typename PolicyFn>
EvalResult evaluate_policy(BookFactory make_book, PolicyFn policy, int n_episodes,
                            int64_t n_events, uint64_t base_seed) {
    std::vector<double> norm_pnls, maps;
    for (int i = 0; i < n_episodes; i++) {
        Rng rng(base_seed + 7919ULL * (i + 1));
        Book book = make_book(rng);
        MarketMakingEnv<Book> env(book, rng, SPOONER_ORDER_SIZE, 0.005, 50, 15, 60, 60, 14,
                                   SPOONER_MIN_INV, SPOONER_MAX_INV);
        State state = env.reset(100.0, 3000, 200);
        double abs_inv_sum = 0.0;
        StepInfo info;
        for (int64_t t = 0; t < n_events; t++) {
            int action = policy(state, rng);
            state = env.step(action, info);
            abs_inv_sum += std::fabs(env.inv());
        }
        double pnl = env.cash() + env.inv() * env.mid();
        norm_pnls.push_back(pnl / (double) n_events);
        maps.push_back(abs_inv_sum / (double) n_events);
    }
    double mean = 0.0;
    for (double v : norm_pnls) mean += v;
    mean /= norm_pnls.size();
    double var = 0.0;
    for (double v : norm_pnls) var += (v - mean) * (v - mean);
    var /= norm_pnls.size();
    double map_mean = 0.0;
    for (double v : maps) map_mean += v;
    map_mean /= maps.size();
    return EvalResult{mean, std::sqrt(var), map_mean};
}

int main(int argc, char** argv) {
    std::string weights_path;
    std::string book_kind = "qr";
    int episodes = 20;
    int64_t events = 17'600'000;
    std::string policy_arg = "all";
    std::string cfg_dir = ".";
    std::string state_kind = "full";  // must match how the loaded weights were trained

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--weights") weights_path = next();
        else if (a == "--book") book_kind = next();
        else if (a == "--episodes") episodes = std::stoi(next());
        else if (a == "--events") events = std::stoll(next());
        else if (a == "--policy") policy_arg = next();
        else if (a == "--cfg") cfg_dir = next();
        else if (a == "--state") state_kind = next();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    bool basic_state = (state_kind == "basic");

    HashingTileCoder ac(N_AGENT_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 0, TILE_SEED_AGENT);
    HashingTileCoder mc(N_MARKET_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 1, TILE_SEED_MARKET);
    HashingTileCoder fc(N_FULL_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 2, TILE_SEED_FULL);

    std::vector<float> weights;
    if (!weights_path.empty()) {
        weights = load_weights(weights_path);
        std::printf("Loaded weights from %s (%zu floats)\n", weights_path.c_str(), weights.size());
    }

    auto run_for_book = [&](auto make_book_template) {
        using Book = decltype(make_book_template(*(Rng*) nullptr));
        auto run = [&](const char* label, auto policy) {
            EvalResult r = evaluate_policy<Book>(make_book_template, policy, episodes, events, EVAL_BASE_SEED);
            std::printf("  %-12s norm_pnl_mean=%.6f norm_pnl_std=%.6f map_mean=%.2f\n",
                        label, r.norm_pnl_mean, r.norm_pnl_std, r.map_mean);
        };

        if (!weights.empty() && (policy_arg == "greedy" || policy_arg == "all")) {
            SpoonerAgent agent(weights.data(), (int64_t) weights.size(), ac, mc, fc, 999, basic_state);
            run("greedy", [&](const State& s, Rng&) {
                double qs[N_ACTIONS];
                ActionIndices idx;
                agent.q_all_actions(s, qs, idx);
                int best = 0; double bestq = qs[0];
                for (int a = 1; a < N_ACTIONS; a++) if (qs[a] > bestq) { bestq = qs[a]; best = a; }
                return best;
            });
        }
        if (policy_arg == "fixed" || policy_arg == "all") {
            for (int theta = 1; theta <= 5; theta++) {
                int action = -1;
                for (int a = 0; a < N_ACTIONS - 1; a++)
                    if (ACTION_THETA[a].first == theta && ACTION_THETA[a].second == theta) action = a;
                char label[32];
                std::snprintf(label, sizeof(label), "fixed_t%d", theta);
                run(label, [action](const State&, Rng&) { return action; });
            }
        }
        if (policy_arg == "random" || policy_arg == "all") {
            run("random", [](const State&, Rng& rng) { return rng.uniform_int(0, N_ACTIONS); });
        }
    };

    std::printf("=== %s simulator, %d episodes x %lld events, state=%s ===\n",
                book_kind.c_str(), episodes, (long long) events, state_kind.c_str());
    if (book_kind == "qr") {
        QRParams qp = load_qr_params(cfg_dir + "/qr_params.txt");
        auto trade_sizes = std::make_shared<std::vector<double>>(load_trade_sizes(cfg_dir + "/trade_sizes.txt"));
        run_for_book([qp, trade_sizes](Rng& rng) { return QueueReactiveBook(qp, trade_sizes.get(), rng); });
    } else {
        ZIParams zp = load_zi_params(cfg_dir + "/zi_params.txt");
        run_for_book([zp](Rng& rng) { return ZeroIntelligenceBook(zp, rng, 200); });
    }
    return 0;
}
