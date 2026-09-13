// Environment wrapper -- port of env.py's MarketMakingEnv. Templated on
// the book type (QueueReactiveBook or ZeroIntelligenceBook) so both plug
// in with zero virtual-dispatch cost, matching the actual repo's
// Intraday<T1,T2> template pattern. Ported mechanically from env.py --
// see that file's docstring for the *why* behind each design choice
// (agent orders as an "ahead"/"remaining" overlay never touching the
// background simulator's own arrays, FIFO fill resolution, etc.); this
// file preserves every one of those behaviors exactly, including a couple
// of subtleties worth flagging:
//   - prev_mid_ is updated ONLY once per step(), after computing delta_m --
//     never inside update_market_state() and never during reset()'s
//     warmup loop. That means d_mid computed during warmup is always
//     "current mid minus the ORIGINAL reset mid", not a step-over-step
//     delta. This matches env.py's actual (if slightly surprising)
//     behavior exactly, not a "fixed" version of it -- faithfulness to
//     the already-verified Python implementation matters more here than
//     whether that particular warmup quirk looks intentional.
//   - Event::side for MO events is pre-resolved to the CONSUMED side
//     (ASK for a buy MO, BID for a sell MO) by the book itself -- see
//     market_common.hpp -- skipping the 'buy'->'ask' translation env.py
//     does inline.
//   - Reward: RewardMode::ASYM_DAMPED (Eq. 6, eta=0.6, the consolidated
//     agent's reward) is the default; RewardMode::PNL (undamped, the
//     basic agent's reward) is also implemented for the basic-vs-
//     consolidated comparison. env.py's third mode, 'sym_damped', was
//     never used by any variant actually trained here, so it's the one
//     piece of env.py's reward logic not ported.
#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <utility>
#include "market_common.hpp"
#include "rng.hpp"

constexpr int N_ACTIONS = 10;
constexpr int MO_CLEAR_ACTION = 9;
constexpr std::array<std::pair<int, int>, N_ACTIONS - 1> ACTION_THETA = {{
    {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
    {1, 3}, {3, 1}, {2, 5}, {5, 2},
}};

class RollingWindow {
public:
    explicit RollingWindow(int length) : length_(length), buf_(length, 0.0) {}

    void push(double x) {
        if (count_ < length_) {
            buf_[head_] = x;
            sum_ += x;
            sumsq_ += x * x;
            head_ = (head_ + 1) % length_;
            count_++;
        } else {
            double old = buf_[head_];
            sum_ += x - old;
            sumsq_ += x * x - old * old;
            buf_[head_] = x;
            head_ = (head_ + 1) % length_;
        }
    }

    double mean() const { return count_ > 0 ? sum_ / count_ : 0.0; }

    double stddev() const {
        if (count_ <= 1) return 0.0;
        double m = mean();
        double var = sumsq_ / count_ - m * m;
        return std::sqrt(std::max(0.0, var));
    }

    void clear() {
        std::fill(buf_.begin(), buf_.end(), 0.0);
        count_ = 0; head_ = 0; sum_ = 0.0; sumsq_ = 0.0;
    }

private:
    int length_;
    std::vector<double> buf_;
    int count_ = 0, head_ = 0;
    double sum_ = 0.0, sumsq_ = 0.0;
};

struct AgentOrder {
    double price = 0.0, ahead = 0.0, remaining = 0.0;
    bool active = false;
};

struct State {
    double inventory, theta_a, theta_b;
    double spread, mid_price_move, imbalance, signed_volume, volatility, rsi;
};

enum class RewardMode { PNL, ASYM_DAMPED };

struct StepInfo {
    Event event;
    double dt = 0.0;
    double matched_a = 0.0, matched_b = 0.0;
    double psi = 0.0, mid = 0.0;
    int action_taken = 0, action_requested = 0;
    double reward = 0.0;
};

template <typename Book>
class MarketMakingEnv {
public:
    MarketMakingEnv(Book& book, Rng& rng, double order_size = 1000.0, double tick_size = 0.005,
                     int spread_lookback = 50, int mpm_lookback = 15, int vol_lookback = 60,
                     int svl_lookback = 60, int rsi_period = 14,
                     double min_inv = -10000.0, double max_inv = 10000.0,
                     double mo_clear_alpha = 1.0, double damping_eta = 0.6,
                     RewardMode reward_mode = RewardMode::ASYM_DAMPED)
        : book_(book), rng_(rng), order_size_(order_size), tick_size_(tick_size),
          min_inv_(min_inv), max_inv_(max_inv), mo_clear_alpha_(mo_clear_alpha), eta_(damping_eta),
          reward_mode_(reward_mode),
          half_spread_win_(spread_lookback), mpm_win_(mpm_lookback), ret_win_(vol_lookback),
          svl_win_(svl_lookback), rsi_gains_(rsi_period), rsi_losses_(rsi_period) {}

    State reset(double mid_price, int book_burn_in, int warmup_steps) {
        Snapshot snap = book_.reset(mid_price, book_burn_in);
        inv_ = 0.0; cash_ = 0.0; prev_mid_ = snap.mid;
        agent_bid_.active = false; agent_ask_.active = false;
        theta_a_ = 1; theta_b_ = 1;
        half_spread_win_.clear(); mpm_win_.clear(); ret_win_.clear(); svl_win_.clear();
        rsi_gains_.clear(); rsi_losses_.clear();
        half_spread_win_.push(snap.spread / 2.0);
        for (int i = 0; i < warmup_steps; i++) {
            Event ev;
            book_.step_background(ev);
            update_market_state(ev);
        }
        return get_state();
    }

    State step(int action, StepInfo& info) {
        int requested = action;
        if (inv_ >= max_inv_ || inv_ <= min_inv_) action = MO_CLEAR_ACTION;

        if (action == MO_CLEAR_ACTION) {
            agent_bid_.active = false; agent_ask_.active = false;
            double fill_qty = 0.0, avg_price = 0.0;
            execute_agent_market_order(fill_qty, avg_price);
            if (fill_qty != 0.0) {
                inv_ += fill_qty;
                cash_ -= fill_qty * avg_price;
            }
        } else {
            int ta = ACTION_THETA[action].first, tb = ACTION_THETA[action].second;
            theta_a_ = ta; theta_b_ = tb;
            place_agent_orders(ta, tb);
        }

        Snapshot pre_snap = book_.snapshot();
        Event event;
        double dt = book_.step_background(event);
        double matched_a, matched_b, p_a, p_b;
        resolve_fills(pre_snap, event, matched_a, matched_b, p_a, p_b);
        update_market_state(event);  // prev_mid_ still OLD here -- see file header note

        Snapshot post = book_.snapshot();
        double mid = post.mid;
        double delta_m = mid - prev_mid_;
        prev_mid_ = mid;

        double psi_a = matched_a > 0.0 ? matched_a * (p_a - mid) : 0.0;
        double psi_b = matched_b > 0.0 ? matched_b * (mid - p_b) : 0.0;
        double psi = psi_a + psi_b + inv_ * delta_m;
        double reward = (reward_mode_ == RewardMode::PNL)
            ? psi
            : psi - std::max(0.0, eta_ * inv_ * delta_m);

        info.event = event; info.dt = dt; info.matched_a = matched_a; info.matched_b = matched_b;
        info.psi = psi; info.mid = mid; info.action_taken = action; info.action_requested = requested;
        info.reward = reward;
        return get_state();
    }

    double inv() const { return inv_; }
    double cash() const { return cash_; }
    double mid() const { return book_.snapshot().mid; }
    double min_inv() const { return min_inv_; }
    double max_inv() const { return max_inv_; }

    // ---- test-only accessors (unit tests need to poke/peek at private
    // agent-order overlay state directly, matching test_env.py's use of
    // env._new_agent_order/_resolve_fills/env.agent_bid). Zero cost to
    // production code -- nothing else calls these. ----
    void test_set_agent_bid(double price, double ahead, double remaining) {
        agent_bid_ = AgentOrder{price, ahead, remaining, true};
    }
    void test_set_agent_ask(double price, double ahead, double remaining) {
        agent_ask_ = AgentOrder{price, ahead, remaining, true};
    }
    AgentOrder test_get_agent_bid() const { return agent_bid_; }
    AgentOrder test_get_agent_ask() const { return agent_ask_; }
    void test_resolve_fills(const Snapshot& pre_snap, const Event& event,
                             double& matched_a, double& matched_b, double& p_a, double& p_b) {
        resolve_fills(pre_snap, event, matched_a, matched_b, p_a, p_b);
    }
    AgentOrder test_new_agent_order(bool is_ask, double price, const Snapshot& snap) const {
        return new_agent_order(is_ask, price, snap);
    }

private:
    static double round_to_tick(double price, double tick) {
        return std::round(price / tick) * tick;
    }

    void place_agent_orders(int theta_a, int theta_b) {
        Snapshot snap = book_.snapshot();
        double ref = snap.mid;
        double spread_scale = half_spread_win_.mean();
        if (spread_scale <= 0.0) spread_scale = snap.spread / 2.0;
        double ask_price = round_to_tick(ref + theta_a * spread_scale, tick_size_);
        double bid_price = round_to_tick(ref - theta_b * spread_scale, tick_size_);
        ask_price = std::max(ask_price, snap.bid_prices[0] + tick_size_);
        bid_price = std::min(bid_price, snap.ask_prices[0] - tick_size_);

        if (!agent_ask_.active || std::fabs(agent_ask_.price - ask_price) > tick_size_ / 2.0)
            agent_ask_ = new_agent_order(true, ask_price, snap);
        if (!agent_bid_.active || std::fabs(agent_bid_.price - bid_price) > tick_size_ / 2.0)
            agent_bid_ = new_agent_order(false, bid_price, snap);
    }

    AgentOrder new_agent_order(bool is_ask, double price, const Snapshot& snap) const {
        const auto& prices = is_ask ? snap.ask_prices : snap.bid_prices;
        const auto& sizes = is_ask ? snap.ask_sizes : snap.bid_sizes;
        double ahead = 0.0;
        for (int i = 0; i < N_LEVELS; i++) {
            if (std::fabs(prices[i] - price) <= tick_size_ / 2.0) { ahead = sizes[i]; break; }
        }
        return AgentOrder{price, ahead, order_size_, true};
    }

    void execute_agent_market_order(double& fill_qty, double& avg_price) {
        fill_qty = 0.0; avg_price = 0.0;
        if (inv_ == 0.0) return;
        double size = -mo_clear_alpha_ * inv_;
        Snapshot snap = book_.snapshot();
        const auto& prices = size > 0.0 ? snap.ask_prices : snap.bid_prices;
        const auto& sizes = size > 0.0 ? snap.ask_sizes : snap.bid_sizes;
        double remaining = std::fabs(size);
        double cost = 0.0, filled = 0.0;
        for (int i = 0; i < N_LEVELS; i++) {
            double take = std::min(remaining, sizes[i]);
            if (take <= 0.0) continue;
            cost += take * prices[i];
            filled += take;
            remaining -= take;
            if (remaining <= 0.0) break;
        }
        if (filled == 0.0) return;
        avg_price = cost / filled;
        fill_qty = (size > 0.0) ? filled : -filled;
    }

    void resolve_fills(const Snapshot& pre_snap, const Event& event,
                        double& matched_a, double& matched_b, double& p_a, double& p_b) {
        matched_a = matched_b = 0.0;
        p_a = agent_ask_.active ? agent_ask_.price : 0.0;
        p_b = agent_bid_.active ? agent_bid_.price : 0.0;

        if (event.type == EventType::MO) {
            bool consumed_ask = (event.side == BookSide::ASK);
            AgentOrder* agent_order = consumed_ask
                ? (agent_ask_.active ? &agent_ask_ : nullptr)
                : (agent_bid_.active ? &agent_bid_ : nullptr);
            const auto& prices = consumed_ask ? pre_snap.ask_prices : pre_snap.bid_prices;
            const auto& sizes = consumed_ask ? pre_snap.ask_sizes : pre_snap.bid_sizes;
            double remaining = event.size;
            for (int i = 0; i < N_LEVELS && remaining > 0.0; i++) {
                double p = prices[i], s = sizes[i];
                bool is_agent_level = (agent_order != nullptr &&
                                        std::fabs(p - agent_order->price) <= tick_size_ / 2.0);
                if (is_agent_level) {
                    double ahead_before = agent_order->ahead;
                    double behind_before = std::max(0.0, s - ahead_before);
                    double eat_ahead = std::min(ahead_before, remaining);
                    agent_order->ahead = ahead_before - eat_ahead;
                    remaining -= eat_ahead;
                    double fill = std::min(agent_order->remaining, remaining);
                    agent_order->remaining -= fill;
                    remaining -= fill;
                    if (fill > 0.0) {
                        if (consumed_ask) matched_a += fill; else matched_b += fill;
                    }
                    remaining -= std::min(behind_before, remaining);
                    if (agent_order->remaining <= 0.0) {
                        if (consumed_ask) agent_ask_.active = false; else agent_bid_.active = false;
                        agent_order = nullptr;
                    }
                } else {
                    remaining -= std::min(s, remaining);
                }
            }
            if (matched_a > 0.0) { inv_ -= matched_a; cash_ += matched_a * p_a; }
            if (matched_b > 0.0) { inv_ += matched_b; cash_ -= matched_b * p_b; }
            return;
        }

        if (event.type == EventType::CANCEL) {
            bool ask_side = (event.side == BookSide::ASK);
            const auto& prices = ask_side ? pre_snap.ask_prices : pre_snap.bid_prices;
            const auto& sizes = ask_side ? pre_snap.ask_sizes : pre_snap.bid_sizes;
            int level = event.level;
            if (level < N_LEVELS) {
                double price = prices[level], size_before = sizes[level], qty = event.size;
                if (ask_side && agent_ask_.active && std::fabs(price - agent_ask_.price) <= tick_size_ / 2.0)
                    apply_cancel_to_agent_order(agent_ask_, qty, size_before);
                if (!ask_side && agent_bid_.active && std::fabs(price - agent_bid_.price) <= tick_size_ / 2.0)
                    apply_cancel_to_agent_order(agent_bid_, qty, size_before);
            }
        }
    }

    void apply_cancel_to_agent_order(AgentOrder& order, double qty, double level_size_before) {
        if (level_size_before <= 0.0) return;
        double p_ahead = std::min(1.0, order.ahead / level_size_before);
        if (rng_.uniform() < p_ahead) order.ahead = std::max(0.0, order.ahead - qty);
    }

    void update_market_state(const Event& event) {
        Snapshot snap = book_.snapshot();
        half_spread_win_.push(snap.spread / 2.0);
        double mid = snap.mid;
        double d_mid = mid - prev_mid_;
        mpm_win_.push(d_mid);
        ret_win_.push(d_mid);
        double signed_vol = 0.0;
        if (event.type == EventType::MO)
            signed_vol = (event.side == BookSide::ASK) ? event.size : -event.size;
        svl_win_.push(signed_vol);
        rsi_gains_.push(std::max(0.0, d_mid));
        rsi_losses_.push(std::max(0.0, -d_mid));
    }

    double rsi() const {
        double avg_gain = rsi_gains_.mean();
        double avg_loss = rsi_losses_.mean();
        if (avg_loss == 0.0) return avg_gain > 0.0 ? 100.0 : 50.0;
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    static double imbalance(const Snapshot& snap) {
        double bid_depth = 0.0, ask_depth = 0.0;
        for (int i = 0; i < N_LEVELS; i++) { bid_depth += snap.bid_sizes[i]; ask_depth += snap.ask_sizes[i]; }
        double total = bid_depth + ask_depth;
        return total > 0.0 ? (bid_depth - ask_depth) / total : 0.0;
    }

    State get_state() const {
        Snapshot snap = book_.snapshot();
        State s;
        s.inventory = inv_; s.theta_a = theta_a_; s.theta_b = theta_b_;
        s.spread = snap.spread; s.mid_price_move = mpm_win_.mean();
        s.imbalance = imbalance(snap); s.signed_volume = svl_win_.mean();
        s.volatility = ret_win_.stddev(); s.rsi = rsi();
        return s;
    }

    Book& book_;
    Rng& rng_;
    double order_size_, tick_size_;
    double min_inv_, max_inv_, mo_clear_alpha_, eta_;
    RewardMode reward_mode_;

    double inv_ = 0.0, cash_ = 0.0, prev_mid_ = 0.0;
    AgentOrder agent_bid_, agent_ask_;
    int theta_a_ = 1, theta_b_ = 1;

    RollingWindow half_spread_win_, mpm_win_, ret_win_, svl_win_, rsi_gains_, rsi_losses_;
};
