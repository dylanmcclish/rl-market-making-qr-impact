// Part 5 training & evaluation, C++ port of train.py -- Hogwild-style
// multi-threaded training (std::thread, not std::mutex, around the actual
// weight update -- matching main.cpp's train() in the real repo, and
// genuinely simpler here than the Python version's multiprocessing.RawArray
// dance, since std::thread already shares process memory natively).
// Only the episode counter is synchronized (atomic fetch_add), same as
// their episode_mutex.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "agent.hpp"
#include "env.hpp"
#include "market_qr.hpp"
#include "market_zi.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "tile_coding.hpp"

constexpr int64_t EVENTS_PER_DAY = 17'600'000;  // measured, see cpp/README notes
constexpr uint64_t TILE_SEED_AGENT = 0, TILE_SEED_MARKET = 1, TILE_SEED_FULL = 2;

struct EpisodeStats {
    int episode, worker;
    double reward, pnl;
    int64_t n_events, n_fills;
    double mean_abs_inv, final_inv, wall_seconds;
};

std::mutex g_print_mutex;
std::mutex g_csv_mutex;

template <typename Book, typename BookFactory>
EpisodeStats run_training_episode(SpoonerAgent& agent, BookFactory make_book, Rng& episode_rng,
                                   int episode_idx, int64_t n_events, int worker_id,
                                   RewardMode reward_mode, double eps_t) {
    Book book = make_book(episode_rng);
    MarketMakingEnv<Book> env(book, episode_rng, SPOONER_ORDER_SIZE, 0.005, 50, 15, 60, 60, 14,
                               SPOONER_MIN_INV, SPOONER_MAX_INV, 1.0, 0.6, reward_mode);
    State state = env.reset(100.0, 3000, 200);

    ActionIndices cur_idx;
    double cur_qs[N_ACTIONS];
    agent.q_all_actions(state, cur_qs, cur_idx);
    int action = agent.act_from_qs(cur_qs, (double) episode_idx, eps_t);

    double total_reward = 0.0;
    int64_t n_fills = 0;
    double abs_inv_sum = 0.0;
    StepInfo info;

    for (int64_t t = 0; t < n_events; t++) {
        State next_state = env.step(action, info);
        int actual_action = info.action_taken;

        ActionIndices next_idx;
        double next_qs[N_ACTIONS];
        agent.q_all_actions(next_state, next_qs, next_idx);
        int next_action = agent.act_from_qs(next_qs, (double) episode_idx, eps_t);
        double next_q = next_qs[next_action];

        agent.update(cur_idx, actual_action, info.reward, next_q, false);

        if (info.matched_a > 0.0 || info.matched_b > 0.0) n_fills++;
        total_reward += info.reward;
        abs_inv_sum += std::fabs(env.inv());

        cur_idx = next_idx;
        action = next_action;
    }
    agent.clear_traces();

    EpisodeStats stats{};
    stats.episode = episode_idx;
    stats.worker = worker_id;
    stats.reward = total_reward;
    stats.pnl = env.cash() + env.inv() * env.mid();
    stats.n_events = n_events;
    stats.n_fills = n_fills;
    stats.mean_abs_inv = abs_inv_sum / (double) n_events;
    stats.final_inv = env.inv();
    return stats;
}

template <typename Book, typename BookFactory>
void worker_loop(int worker_id, float* theta, const HashingTileCoder& ac,
                  const HashingTileCoder& mc, const HashingTileCoder& fc,
                  BookFactory make_book, std::atomic<int>& episode_counter, int n_episodes,
                  int64_t n_events, uint64_t base_seed, std::ofstream& csv,
                  bool basic_state, RewardMode reward_mode, double eps_t) {
    Rng agent_rng(base_seed + 1000ULL * worker_id + 1);
    SpoonerAgent agent(theta, MEMORY_SIZE, ac, mc, fc, base_seed + 1000ULL * worker_id + 1, basic_state);

    while (true) {
        int episode_idx = episode_counter.fetch_add(1);
        if (episode_idx >= n_episodes) break;

        Rng episode_rng(base_seed + 7919ULL * (episode_idx + 1));
        auto t0 = std::chrono::steady_clock::now();
        EpisodeStats stats = run_training_episode<Book>(agent, make_book, episode_rng,
                                                          episode_idx, n_events, worker_id,
                                                          reward_mode, eps_t);
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        stats.wall_seconds = wall;

        {
            std::lock_guard<std::mutex> lk(g_csv_mutex);
            csv << stats.episode << ',' << stats.worker << ',' << stats.reward << ','
                << stats.pnl << ',' << stats.n_events << ',' << stats.n_fills << ','
                << stats.mean_abs_inv << ',' << stats.final_inv << ',' << stats.wall_seconds << '\n';
            csv.flush();
        }
        {
            std::lock_guard<std::mutex> lk(g_print_mutex);
            std::printf("[ep %d] worker=%d reward=%.2f pnl=%.2f fills=%lld wall=%.1fs\n",
                        stats.episode, stats.worker, stats.reward, stats.pnl,
                        (long long) stats.n_fills, stats.wall_seconds);
        }
    }
}

template <typename Book, typename BookFactory>
void train_hogwild(BookFactory make_book, const std::string& out_tag, int n_episodes, int n_workers,
                    int64_t n_events, uint64_t base_seed, const std::string& out_dir,
                    bool basic_state, RewardMode reward_mode, double eps_t) {
    std::vector<float> theta(MEMORY_SIZE, 0.0f);
    HashingTileCoder ac(N_AGENT_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 0, TILE_SEED_AGENT);
    HashingTileCoder mc(N_MARKET_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 1, TILE_SEED_MARKET);
    HashingTileCoder fc(N_FULL_VARS, N_TILINGS, TILES_PER_DIM, MEMORY_SIZE, 2, TILE_SEED_FULL);

    std::atomic<int> episode_counter{0};
    std::string csv_path = out_dir + "/training_log_" + out_tag + ".cpp.csv";
    std::ofstream csv(csv_path);
    csv << "episode,worker,reward,pnl,n_events,n_fills,mean_abs_inv,final_inv,wall_seconds\n";

    auto t_start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int w = 0; w < n_workers; w++) {
        threads.emplace_back(worker_loop<Book, BookFactory>, w, theta.data(), std::cref(ac),
                              std::cref(mc), std::cref(fc), make_book, std::ref(episode_counter),
                              n_episodes, n_events, base_seed, std::ref(csv),
                              basic_state, reward_mode, eps_t);
    }
    for (auto& t : threads) t.join();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("Training done: %d episodes, %d workers, %.1f min elapsed\n",
                n_episodes, n_workers, elapsed / 60.0);

    std::string w_path = out_dir + "/trained_weights_" + out_tag + ".cpp.bin";
    std::ofstream wf(w_path, std::ios::binary);
    wf.write(reinterpret_cast<const char*>(theta.data()), (std::streamsize) (theta.size() * sizeof(float)));
    std::printf("Saved trained weights -> %s\n", w_path.c_str());
}

// ---------------------------------------------------------------------- //
// CLI
// ---------------------------------------------------------------------- //
int main(int argc, char** argv) {
    std::string mode = "smoke";
    int episodes = 1000;
    int workers = (int) std::max(1u, std::thread::hardware_concurrency() - 1);
    std::string book_kind = "qr";
    int64_t events = EVENTS_PER_DAY;
    std::string out_dir = ".";
    uint64_t base_seed = 0;
    std::string state_kind = "full";     // "full" (LCTC, consolidated) or "basic" (agent-only)
    std::string reward_kind = "asym";    // "asym" (damped, consolidated) or "pnl" (undamped, basic)
    double eps_t = EPS_T;                // Table 2 default 1000; override for rescaled schedules
    std::string tag;                     // output-filename tag; defaults to "<book>_<agent>"

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--episodes") episodes = std::stoi(next());
        else if (a == "--workers") workers = std::stoi(next());
        else if (a == "--book") book_kind = next();
        else if (a == "--events") events = std::stoll(next());
        else if (a == "--out") out_dir = next();
        else if (a == "--seed") base_seed = std::stoull(next());
        else if (a == "--state") state_kind = next();
        else if (a == "--reward") reward_kind = next();
        else if (a == "--eps-t") eps_t = std::stod(next());
        else if (a == "--tag") tag = next();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    if (mode == "smoke") {
        episodes = workers * 3;
        events = 4000;
    }

    bool basic_state = (state_kind == "basic");
    if (!basic_state && state_kind != "full") {
        std::fprintf(stderr, "unknown --state %s (expected basic or full)\n", state_kind.c_str());
        return 1;
    }
    RewardMode reward_mode = (reward_kind == "pnl") ? RewardMode::PNL : RewardMode::ASYM_DAMPED;
    if (reward_kind != "pnl" && reward_kind != "asym") {
        std::fprintf(stderr, "unknown --reward %s (expected pnl or asym)\n", reward_kind.c_str());
        return 1;
    }
    std::string agent_kind = basic_state ? "basic" : "consolidated";
    if (tag.empty()) tag = book_kind + "_" + agent_kind;

    std::printf("mode=%s episodes=%d workers=%d book=%s events=%lld state=%s reward=%s eps_t=%.1f tag=%s\n",
                mode.c_str(), episodes, workers, book_kind.c_str(), (long long) events,
                state_kind.c_str(), reward_kind.c_str(), eps_t, tag.c_str());

    if (book_kind == "qr") {
        QRParams qp = load_qr_params(out_dir + "/qr_params.txt");
        auto trade_sizes = std::make_shared<std::vector<double>>(load_trade_sizes(out_dir + "/trade_sizes.txt"));
        auto make_book = [qp, trade_sizes](Rng& rng) {
            return QueueReactiveBook(qp, trade_sizes.get(), rng);
        };
        train_hogwild<QueueReactiveBook>(make_book, tag, episodes, workers, events, base_seed, out_dir,
                                          basic_state, reward_mode, eps_t);
    } else if (book_kind == "zi") {
        ZIParams zp = load_zi_params(out_dir + "/zi_params.txt");
        auto make_book = [zp](Rng& rng) { return ZeroIntelligenceBook(zp, rng, 200); };
        train_hogwild<ZeroIntelligenceBook>(make_book, tag, episodes, workers, events, base_seed, out_dir,
                                             basic_state, reward_mode, eps_t);
    } else {
        std::fprintf(stderr, "unknown --book %s (expected qr or zi)\n", book_kind.c_str());
        return 1;
    }
    return 0;
}
