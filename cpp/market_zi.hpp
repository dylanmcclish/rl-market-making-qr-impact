// Zero-intelligence / Poisson control simulator -- port of market_zi.py's
// ZeroIntelligenceBook. Calibration (alpha/mu/delta/dp/sigma) is done
// once, offline, in Python (cpp/export_config.py) since it requires
// parsing a ~480MB real price-level CSV -- not part of the hot training
// path, so there's no reason to reimplement that parse here. This class
// only ports the simulation itself.
#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include "market_common.hpp"
#include "rng.hpp"

struct ZIParams {
    double alpha, mu, delta, dp, sigma;
};

class ZeroIntelligenceBook {
public:
    ZeroIntelligenceBook(const ZIParams& p, Rng& rng, int n_ticks = 200)
        : alpha_(p.alpha), mu_(p.mu), delta_(p.delta), dp_(p.dp), sigma_(p.sigma),
          n_ticks_(n_ticks), rng_(rng),
          lo_chunk_rate_(p.alpha / p.sigma), mo_chunk_rate_(p.mu / p.sigma / 2.0) {
        bid_book_.assign(n_ticks_, 0.0);
        ask_book_.assign(n_ticks_, 0.0);
    }

    Snapshot reset(double mid_price, int burn_in_events) {
        mid0_ = mid_price;
        double steady_depth = alpha_ / delta_;  // Smith Table III asymptotic depth
        double far = steady_depth * dp_;
        std::fill(bid_book_.begin(), bid_book_.end(), far);
        std::fill(ask_book_.begin(), ask_book_.end(), far);
        for (int i = 0; i < burn_in_events; i++) {
            Event ev;
            step_background(ev);
        }
        return snapshot();
    }

    double step_background(Event& out_event) {
        int bid_touch = touch_index(bid_book_);
        int ask_touch = touch_index(ask_book_);

        double lo_bid_rate = lo_chunk_rate_ * dp_ * n_ticks_;
        double lo_ask_rate = lo_chunk_rate_ * dp_ * n_ticks_;
        double bid_sum = 0.0, ask_sum = 0.0;
        for (double v : bid_book_) bid_sum += v;
        for (double v : ask_book_) ask_sum += v;
        double cancel_bid_total = delta_ * bid_sum / sigma_;
        double cancel_ask_total = delta_ * ask_sum / sigma_;
        double mo_buy_rate = (ask_touch >= 0) ? mo_chunk_rate_ : 0.0;
        double mo_sell_rate = (bid_touch >= 0) ? mo_chunk_rate_ : 0.0;

        double rates[6] = {lo_bid_rate, lo_ask_rate, cancel_bid_total,
                            cancel_ask_total, mo_buy_rate, mo_sell_rate};
        double total = 0.0;
        for (double r : rates) total += r;
        if (total <= 0.0) {
            out_event = Event{EventType::NOOP, BookSide::BID, 0, 0.0};
            return 0.0;
        }

        double dt = rng_.exponential(total);
        double r = rng_.uniform() * total;
        double cum = 0.0;
        int choice = 5;
        for (int i = 0; i < 5; i++) {
            cum += rates[i];
            if (r < cum) { choice = i; break; }
        }

        if (choice == 0) {
            int i = rng_.uniform_int(0, n_ticks_);
            bid_book_[i] += sigma_;
            out_event = Event{EventType::LO, BookSide::BID, i, sigma_};
        } else if (choice == 1) {
            int i = rng_.uniform_int(0, n_ticks_);
            ask_book_[i] += sigma_;
            out_event = Event{EventType::LO, BookSide::ASK, i, sigma_};
        } else if (choice == 2) {
            int i = weighted_level(bid_book_, bid_sum);
            double removed = std::min(sigma_, bid_book_[i]);
            bid_book_[i] -= removed;
            out_event = Event{EventType::CANCEL, BookSide::BID, i, removed};
        } else if (choice == 3) {
            int i = weighted_level(ask_book_, ask_sum);
            double removed = std::min(sigma_, ask_book_[i]);
            ask_book_[i] -= removed;
            out_event = Event{EventType::CANCEL, BookSide::ASK, i, removed};
        } else if (choice == 4) {
            market_order(ask_book_, sigma_);
            out_event = Event{EventType::MO, BookSide::ASK, 0, sigma_};  // buy MO consumes ask
        } else {
            market_order(bid_book_, sigma_);
            out_event = Event{EventType::MO, BookSide::BID, 0, sigma_};  // sell MO consumes bid
        }
        return dt;
    }

    Snapshot snapshot() const {
        int bt = touch_index(bid_book_);
        int at = touch_index(ask_book_);
        Snapshot s;
        int bt0 = bt >= 0 ? bt : 0;
        int at0 = at >= 0 ? at : 0;
        for (int k = 0; k < N_LEVELS; k++) {
            s.bid_prices[k] = mid0_ - dp_ * (k + bt0 + 1);
            s.ask_prices[k] = mid0_ + dp_ * (k + at0 + 1);
            s.bid_sizes[k] = (bt >= 0 && bt0 + k < n_ticks_) ? bid_book_[bt0 + k] : 0.0;
            s.ask_sizes[k] = (at >= 0 && at0 + k < n_ticks_) ? ask_book_[at0 + k] : 0.0;
        }
        if (bt >= 0 && at >= 0) {
            s.spread = s.ask_prices[0] - s.bid_prices[0];
            s.mid = (s.ask_prices[0] + s.bid_prices[0]) / 2.0;
        } else {
            s.spread = 0.0;
            s.mid = mid0_;
        }
        return s;
    }

private:
    static int touch_index(const std::vector<double>& book) {
        for (size_t i = 0; i < book.size(); i++)
            if (book[i] != 0.0) return (int) i;
        return -1;
    }

    int weighted_level(const std::vector<double>& book, double total) {
        if (total <= 0.0) return 0;
        double r = rng_.uniform() * total;
        double cum = 0.0;
        for (size_t i = 0; i < book.size(); i++) {
            cum += book[i];
            if (r < cum) return (int) i;
        }
        return (int) book.size() - 1;
    }

    void market_order(std::vector<double>& book, double size) {
        double remaining = size;
        for (size_t i = 0; i < book.size() && remaining > 0.0; i++) {
            if (book[i] > 0.0) {
                double take = std::min(book[i], remaining);
                book[i] -= take;
                remaining -= take;
            }
        }
    }

    double alpha_, mu_, delta_, dp_, sigma_;
    int n_ticks_;
    Rng& rng_;
    double lo_chunk_rate_, mo_chunk_rate_;
    std::vector<double> bid_book_, ask_book_;
    double mid0_ = 0.0;
};
