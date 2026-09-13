// Small RNG helper wrapping std::mt19937_64 with the specific sampling
// primitives the simulators/agent need (exponential waiting time, weighted
// categorical choice, uniform [0,1), uniform int). Not bit-compatible with
// numpy's PCG64 -- the port is validated behaviorally/statistically against
// the Python simulators (aggressive-fill rates, Smith diagnostics), not by
// reproducing identical random draws -- see cpp/README notes.
#pragma once
#include <random>
#include <vector>
#include <cstdint>

class Rng {
public:
    explicit Rng(uint64_t seed) : eng_(seed), unif_(0.0, 1.0) {}

    double uniform() { return unif_(eng_); }

    double exponential(double rate) {
        // -ln(U)/rate, U in (0,1]
        double u = 1.0 - unif_(eng_);  // avoid log(0)
        return -std::log(u) / rate;
    }

    // Sample an index in [0, n) with probability weights[i]/sum(weights).
    // `weights` need not be normalized; `total` is sum(weights) (caller
    // already has it from building the rate vector, so we take it rather
    // than re-summing).
    int categorical(const double* weights, int n, double total) {
        double r = unif_(eng_) * total;
        double cum = 0.0;
        for (int i = 0; i < n - 1; i++) {
            cum += weights[i];
            if (r < cum) return i;
        }
        return n - 1;
    }

    int uniform_int(int lo_inclusive, int hi_exclusive) {
        std::uniform_int_distribution<int> d(lo_inclusive, hi_exclusive - 1);
        return d(eng_);
    }

    uint64_t raw() { return eng_(); }

private:
    std::mt19937_64 eng_;
    std::uniform_real_distribution<double> unif_;
};
