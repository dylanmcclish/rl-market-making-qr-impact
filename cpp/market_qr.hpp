// Queue-reactive limit order book simulator -- direct port of
// market_qr.py's QueueReactiveBook. See that file's docstring for the
// model itself (state-dependent Poisson intensities for LO/cancel/MO,
// fit to real IEX data); this is a mechanical translation, not a
// re-derivation, so the two should be read side by side for the "why".
#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "market_common.hpp"
#include "rng.hpp"

struct QRParams {
    double Q0, a_LO, b_LO, a_C, a_MO, q_max_observed;
};

class QueueReactiveBook {
public:
    QueueReactiveBook(const QRParams& p, const std::vector<double>* mo_size_pool,
                       Rng& rng, double tick_size = 0.005, double order_size = 100.0)
        : Q0_(p.Q0), a_LO_(p.a_LO), neg_slope_(p.b_LO / p.Q0), slope_(p.a_C / p.Q0),
          a_MO_(p.a_MO), q_max_observed_(p.q_max_observed),
          tick_size_(tick_size), order_size_(order_size),
          mo_size_pool_(mo_size_pool), rng_(rng) {}

    double lo_rate(double q) const {
        double qc = std::min(q, q_max_observed_);
        return a_LO_ * std::exp(-neg_slope_ * qc);
    }
    double cancel_rate(double q) const {
        double qc = std::min(q, q_max_observed_);
        return slope_ * qc;
    }

    Snapshot reset(double mid_price, int burn_in_events) {
        bid_price0_ = mid_price - tick_size_;
        ask_price0_ = mid_price + tick_size_;
        bid_sizes_.fill(Q0_);
        ask_sizes_.fill(Q0_);
        for (int i = 0; i < burn_in_events; i++) {
            Event ev;
            step_background(ev);
        }
        return snapshot();
    }

    // Advances by one background event; fills `out_event` and returns dt.
    double step_background(Event& out_event) {
        double lo_bid[N_LEVELS], lo_ask[N_LEVELS], c_bid[N_LEVELS], c_ask[N_LEVELS];
        double total = 0.0;
        for (int i = 0; i < N_LEVELS; i++) {
            lo_bid[i] = lo_rate(bid_sizes_[i]); total += lo_bid[i];
            lo_ask[i] = lo_rate(ask_sizes_[i]); total += lo_ask[i];
            c_bid[i] = bid_sizes_[i] > 0 ? cancel_rate(bid_sizes_[i]) : 0.0; total += c_bid[i];
            c_ask[i] = ask_sizes_[i] > 0 ? cancel_rate(ask_sizes_[i]) : 0.0; total += c_ask[i];
        }
        double mo_buy = a_MO_ / 2.0, mo_sell = a_MO_ / 2.0;
        total += mo_buy + mo_sell;

        if (total <= 0.0) {
            bid_sizes_[0] += order_size_;
            ask_sizes_[0] += order_size_;
            out_event = Event{EventType::RECOVER, BookSide::BID, 0, 0.0};
            return 0.0;
        }

        double dt = rng_.exponential(total);
        double r = rng_.uniform() * total;
        double cum = 0.0;
        int idx = -1;
        // Same flat ordering as the Python rates concat: lo_bid[5], lo_ask[5],
        // c_bid[5], c_ask[5], mo_buy, mo_sell (21 entries total).
        double flat[4 * N_LEVELS + 2];
        for (int i = 0; i < N_LEVELS; i++) flat[i] = lo_bid[i];
        for (int i = 0; i < N_LEVELS; i++) flat[N_LEVELS + i] = lo_ask[i];
        for (int i = 0; i < N_LEVELS; i++) flat[2 * N_LEVELS + i] = c_bid[i];
        for (int i = 0; i < N_LEVELS; i++) flat[3 * N_LEVELS + i] = c_ask[i];
        flat[4 * N_LEVELS] = mo_buy;
        flat[4 * N_LEVELS + 1] = mo_sell;
        int n_flat = 4 * N_LEVELS + 2;
        for (int i = 0; i < n_flat - 1; i++) {
            cum += flat[i];
            if (r < cum) { idx = i; break; }
        }
        if (idx == -1) idx = n_flat - 1;

        int n = N_LEVELS;
        if (idx < n) {
            int level = idx;
            bid_sizes_[level] += order_size_;
            out_event = Event{EventType::LO, BookSide::BID, level, order_size_};
        } else if (idx < 2 * n) {
            int level = idx - n;
            ask_sizes_[level] += order_size_;
            out_event = Event{EventType::LO, BookSide::ASK, level, order_size_};
        } else if (idx < 3 * n) {
            int level = idx - 2 * n;
            double removed = std::min(order_size_, bid_sizes_[level]);
            bid_sizes_[level] -= removed;
            collapse_if_touch_empty(BookSide::BID);
            out_event = Event{EventType::CANCEL, BookSide::BID, level, removed};
        } else if (idx < 4 * n) {
            int level = idx - 3 * n;
            double removed = std::min(order_size_, ask_sizes_[level]);
            ask_sizes_[level] -= removed;
            collapse_if_touch_empty(BookSide::ASK);
            out_event = Event{EventType::CANCEL, BookSide::ASK, level, removed};
        } else if (idx == 4 * n) {
            double size = draw_mo_size();
            apply_market_order(BookSide::ASK, size);  // buy MO consumes ask liquidity
            out_event = Event{EventType::MO, BookSide::ASK, 0, size};
        } else {
            double size = draw_mo_size();
            apply_market_order(BookSide::BID, size);  // sell MO consumes bid liquidity
            out_event = Event{EventType::MO, BookSide::BID, 0, size};
        }
        return dt;
    }

    Snapshot snapshot() const {
        Snapshot s;
        for (int i = 0; i < N_LEVELS; i++) {
            s.bid_prices[i] = bid_price0_ - tick_size_ * i;
            s.ask_prices[i] = ask_price0_ + tick_size_ * i;
        }
        s.bid_sizes = bid_sizes_;
        s.ask_sizes = ask_sizes_;
        s.mid = (s.bid_prices[0] + s.ask_prices[0]) / 2.0;
        s.spread = s.ask_prices[0] - s.bid_prices[0];
        return s;
    }

private:
    double draw_mo_size() {
        if (mo_size_pool_ != nullptr && !mo_size_pool_->empty()) {
            int i = rng_.uniform_int(0, (int) mo_size_pool_->size());
            return (*mo_size_pool_)[i];
        }
        return order_size_;
    }

    // Bounded to N_LEVELS-1 shifts (see market_qr.py's identical bound and
    // its comment on why an unbounded while here can spin forever).
    void collapse_if_touch_empty(BookSide side) {
        auto& sizes = (side == BookSide::BID) ? bid_sizes_ : ask_sizes_;
        int shifts = 0;
        while (sizes[0] <= 0.0 && shifts < N_LEVELS - 1) {
            for (int i = 0; i < N_LEVELS - 1; i++) sizes[i] = sizes[i + 1];
            sizes[N_LEVELS - 1] = 0.0;
            if (side == BookSide::BID) bid_price0_ -= tick_size_;
            else ask_price0_ += tick_size_;
            shifts++;
        }
        if (sizes[0] <= 0.0) sizes[0] = order_size_;
    }

    void apply_market_order(BookSide resting_side, double size) {
        auto& sizes = (resting_side == BookSide::BID) ? bid_sizes_ : ask_sizes_;
        double remaining = size;
        int level = 0;
        while (remaining > 0.0 && level < N_LEVELS) {
            double avail = sizes[level];
            double take = std::min(avail, remaining);
            if (take > 0.0) {
                sizes[level] -= take;
                remaining -= take;
            }
            if (sizes[level] <= 0.0) {
                collapse_if_touch_empty(resting_side);
                level = 0;
            } else {
                break;
            }
        }
    }

    double Q0_, a_LO_, neg_slope_, slope_, a_MO_, q_max_observed_;
    double tick_size_, order_size_;
    const std::vector<double>* mo_size_pool_;
    Rng& rng_;

    std::array<double, N_LEVELS> bid_sizes_{};
    std::array<double, N_LEVELS> ask_sizes_{};
    double bid_price0_ = 0.0, ask_price0_ = 0.0;
};
