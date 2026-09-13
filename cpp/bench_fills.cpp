// Fill counts for the non-learning benchmark policies (fixed theta=1..5,
// random) -- eval.cpp only aggregates PnL/MAP for these, doesn't log raw
// fill counts, so this is a small standalone companion to diagnose.cpp
// (which covers the trained-agent policies) for an apples-to-apples count.
#include <cstdio>
#include <string>
#include "env.hpp"
#include "market_qr.hpp"
#include "market_zi.hpp"
#include "params.hpp"
#include "rng.hpp"

template <typename Book, typename PolicyFn>
void run(Book& book, PolicyFn policy, int64_t n_events, uint64_t seed, const char* label) {
    Rng rng(seed);
    MarketMakingEnv<Book> env(book, rng, 1000.0, 0.005, 50, 15, 60, 60, 14, -10000.0, 10000.0);
    State state = env.reset(100.0, 3000, 200);
    int64_t n_fills = 0;
    StepInfo info;
    for (int64_t t = 0; t < n_events; t++) {
        int action = policy(state, rng);
        state = env.step(action, info);
        if (info.matched_a > 0.0 || info.matched_b > 0.0) n_fills++;
    }
    std::printf("  %-10s fills=%lld (per %lld events)\n", label, (long long) n_fills, (long long) n_events);
}

int main(int argc, char** argv) {
    std::string book_kind = "qr", cfg_dir = ".";
    int64_t events = 3'000'000;
    uint64_t seed = 424242;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--book") book_kind = next();
        else if (a == "--events") events = std::stoll(next());
        else if (a == "--cfg") cfg_dir = next();
        else if (a == "--seed") seed = std::stoull(next());
    }
    std::printf("=== book=%s events=%lld ===\n", book_kind.c_str(), (long long) events);

    auto fixed_policy = [](int theta) {
        int action = -1;
        for (int a = 0; a < N_ACTIONS - 1; a++)
            if (ACTION_THETA[a].first == theta && ACTION_THETA[a].second == theta) action = a;
        return [action](const State&, Rng&) { return action; };
    };
    auto random_policy = [](const State&, Rng& rng) { return rng.uniform_int(0, N_ACTIONS); };

    if (book_kind == "qr") {
        QRParams qp = load_qr_params(cfg_dir + "/qr_params.txt");
        auto trade_sizes = load_trade_sizes(cfg_dir + "/trade_sizes.txt");
        for (int theta = 1; theta <= 5; theta++) {
            Rng book_rng(seed + 1);
            QueueReactiveBook book(qp, &trade_sizes, book_rng);
            char label[16]; std::snprintf(label, sizeof(label), "theta=%d", theta);
            run(book, fixed_policy(theta), events, seed, label);
        }
        Rng book_rng(seed + 1);
        QueueReactiveBook book(qp, &trade_sizes, book_rng);
        run(book, random_policy, events, seed, "random");
    } else {
        ZIParams zp = load_zi_params(cfg_dir + "/zi_params.txt");
        for (int theta = 1; theta <= 5; theta++) {
            Rng book_rng(seed + 1);
            ZeroIntelligenceBook book(zp, book_rng, 200);
            char label[16]; std::snprintf(label, sizeof(label), "theta=%d", theta);
            run(book, fixed_policy(theta), events, seed, label);
        }
        Rng book_rng(seed + 1);
        ZeroIntelligenceBook book(zp, book_rng, 200);
        run(book, random_policy, events, seed, "random");
    }
    return 0;
}
