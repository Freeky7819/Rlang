#include "rlang.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
namespace rlang {
StepResult step_scalar(const State& s, const Profile& p){
  StepResult out; out.state.phase.resize(s.phase.size()); out.state.amp.resize(s.amp.size());
  for (size_t i=0;i<s.phase.size();++i){
    double delta = p.alpha * std::sin(s.phase[i]*p.omega + p.chord[i]);
    out.state.phase[i] = s.phase[i] + delta;
    out.state.amp[i]   = s.amp[i]   + 0.1*delta;
  }
  out.state.seed = next_seed(s.seed);
  return out;
}
static std::vector<double> parse_line(const std::string& line){
  std::stringstream ss(line); std::vector<double> v; double x; while (ss>>x) v.push_back(x); return v;
}
Profile load_profile_txt(const std::string& path){
  std::ifstream f(path); if(!f) throw std::runtime_error("Cannot open profile txt");
  double alpha, omega; f>>alpha>>omega; std::string line; std::getline(f,line); std::getline(f,line);
  Profile p; p.alpha=alpha; p.omega=omega; p.chord = parse_line(line); return p;
}
State load_state_txt(const std::string& path){
  std::ifstream f(path); if(!f) throw std::runtime_error("Cannot open state txt");
  unsigned long long tmp; f>>tmp; std::string l2,l3; std::getline(f,l2); std::getline(f,l2); std::getline(f,l3);
  State s{}; s.seed=tmp; s.phase = parse_line(l2); s.amp = parse_line(l3);
  if (s.phase.size()!=s.amp.size()) throw std::runtime_error("phase/amp mismatch");
  return s;
}
std::string to_json_like(const StepResult& r){
  std::ostringstream o; o<<std::setprecision(17);
  o << "{\n  \"state\": {\n    \"phase\": [";
  for (size_t i=0;i<r.state.phase.size();++i){ if(i) o<<", "; o<<r.state.phase[i]; }
  o << "],\n    \"amp\": [";
  for (size_t i=0;i<r.state.amp.size();++i){ if(i) o<<", "; o<<r.state.amp[i]; }
  o << "],\n    \"seed\": " << r.state.seed << "\n  }\n}\n";
  return o.str();
}
} // namespace rlang
