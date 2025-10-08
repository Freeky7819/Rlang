#include "rlang.hpp"
#include <cmath>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace rlang {

static uint64_t next_seed(uint64_t seed) {
    return (seed * 6364136223846793005ULL + 1ULL) % ((uint64_t)1 << 64);
}

StepResult step(const State& s, const Profile& p) {
    StepResult out;
    out.state.phase.resize(s.phase.size());
    out.state.amp.resize(s.amp.size());
    double loss = 0.0;
    for (size_t i = 0; i < s.phase.size(); ++i) {
        double delta = p.alpha * std::sin(s.phase[i] * p.omega + p.chord[i]);
        out.state.phase[i] = s.phase[i] + delta;
        out.state.amp[i] = s.amp[i] + delta * 0.1;
        loss += std::abs(s.phase[i] - out.state.phase[i]);
    }
    out.state.seed = next_seed(s.seed);
    out.loss = loss;
    return out;
}

std::string to_json(const StepResult& r) {
    json j;
    j["state"]["phase"] = r.state.phase;
    j["state"]["amp"] = r.state.amp;
    j["state"]["seed"] = r.state.seed;
    j["loss"] = r.loss;
    return j.dump(2);
}

State state_from_json(const std::string& js) {
    auto j = json::parse(js);
    State s;
    s.phase = j["phase"].get<std::vector<double>>();
    s.amp = j["amp"].get<std::vector<double>>();
    s.seed = j["seed"].get<uint64_t>();
    return s;
}

Profile profile_from_json(const std::string& js) {
    auto j = json::parse(js);
    Profile p;
    p.chord = j["chord"].get<std::vector<double>>();
    p.alpha = j["alpha"].get<double>();
    p.omega = j["omega"].get<double>();
    return p;
}

}
