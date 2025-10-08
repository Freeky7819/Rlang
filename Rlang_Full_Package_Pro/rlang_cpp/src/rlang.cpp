#include "rlang.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace rlang {

StepResult step_scalar(const State& s, const Profile& p) {
  StepResult out;
  out.state.phase.resize(s.phase.size());
  out.state.amp.resize(s.amp.size());
  for (size_t i=0;i<s.phase.size();++i){
    double delta = p.alpha * std::sin(s.phase[i]*p.omega + p.chord[i]);
    out.state.phase[i] = s.phase[i] + delta;
    out.state.amp[i]   = s.amp[i]   + 0.1*delta;
  }
  out.state.seed = next_seed(s.seed);
  return out;
}

Profile load_profile_txt(const std::string& path){
  std::ifstream f(path);
  if(!f) throw std::runtime_error("Cannot open profile txt");
  double alpha, omega; std::string chordLine;
  f >> alpha >> omega; std::getline(f, chordLine); // rest of first line
  std::getline(f, chordLine);
  std::stringstream ss(chordLine);
  Profile p; p.alpha=alpha; p.omega=omega;
  double v; while (ss>>v) p.chord.push_back(v);
  return p;
}

State load_state_txt(const std::string& path){
  std::ifstream f(path);
  if(!f) throw std::runtime_error("Cannot open state txt");
  State s{}; unsigned long long tmp; f>>tmp; s.seed=tmp;
  std::string line;
  std::getline(f,line); // endline
  std::getline(f,line); { std::stringstream ss(line); double v; while(ss>>v) s.phase.push_back(v); }
  std::getline(f,line); { std::stringstream ss(line); double v; while(ss>>v) s.amp.push_back(v); }
  if(s.phase.size()!=s.amp.size()) throw std::runtime_error("phase/amp size mismatch");
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
