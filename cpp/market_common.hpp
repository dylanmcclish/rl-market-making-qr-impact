// Shared types for both simulators (market_qr.hpp / market_zi.hpp) and the
// environment (env.hpp), which is templated on the book type so both
// simulators can be dropped in with zero virtual-dispatch cost in the hot
// loop (mirrors the actual tspooner/rl_markets repo's Intraday<T1,T2>
// template pattern).
#pragma once
#include <array>

enum class EventType { LO, CANCEL, MO, RECOVER, NOOP };
enum class BookSide { BID, ASK };

struct Event {
    EventType type = EventType::NOOP;
    // LO/cancel: which side's queue. MO: the side of resting liquidity
    // CONSUMED by the incoming market order -- ASK for a buy MO, BID for a
    // sell MO. (Python's env.py separately derives this from event['side']
    // == 'buy'/'sell'; storing it pre-resolved here skips that redundant
    // translation since nothing else needs the raw aggressor side.)
    BookSide side = BookSide::BID;
    int level = 0;
    double size = 0.0;
};

constexpr int N_LEVELS = 5;

struct Snapshot {
    std::array<double, N_LEVELS> bid_prices{};
    std::array<double, N_LEVELS> bid_sizes{};
    std::array<double, N_LEVELS> ask_prices{};
    std::array<double, N_LEVELS> ask_sizes{};
    double mid = 0.0;
    double spread = 0.0;
};
