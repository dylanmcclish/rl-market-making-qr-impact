// Loads the plain key/value config files cpp/export_config.py writes from
// the already-calibrated Python parameters (calibrated_params_7day.json
// plus the one-time ZI spread calibration, which needs a ~480MB real
// price-level CSV walk -- a one-time offline computation, not part of the
// hot training path, so there's no reason to reimplement that parse
// here). No JSON/CSV parsing needed in C++ at all as a result.
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "market_qr.hpp"
#include "market_zi.hpp"

inline std::unordered_map<std::string, double> load_kv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::unordered_map<std::string, double> m;
    std::string key; double val;
    while (f >> key >> val) m[key] = val;
    return m;
}

inline QRParams load_qr_params(const std::string& path = "qr_params.txt") {
    auto m = load_kv(path);
    return QRParams{m.at("Q0"), m.at("a_LO"), m.at("b_LO"), m.at("a_C"), m.at("a_MO"),
                     m.at("q_max_observed")};
}

inline ZIParams load_zi_params(const std::string& path = "zi_params.txt") {
    auto m = load_kv(path);
    return ZIParams{m.at("alpha"), m.at("mu"), m.at("delta"), m.at("dp"), m.at("sigma")};
}

inline std::vector<double> load_trade_sizes(const std::string& path = "trade_sizes.txt") {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<double> sizes;
    double v;
    while (f >> v) sizes.push_back(v);
    return sizes;
}
