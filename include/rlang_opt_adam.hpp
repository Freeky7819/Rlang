
#pragma once
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

namespace rlangf_opt {

// Adam over numeric gradients on current state (black-box, lightweight).
inline void adam(std::vector<double>& params,
                 std::function<void(const std::vector<double>&)> apply,
                 std::function<double()> loss,
                 int iters=200, double lr=0.05, double eps=1e-8, double beta1=0.9, double beta2=0.999, double fd_eps=1e-3){
    const int D = (int)params.size();
    std::vector<double> m(D,0.0), v(D,0.0);
    auto grad = [&](std::vector<double>& g){
        g.assign(D,0.0);
        double L0 = loss();
        for(int d=0; d<D; ++d){
            auto p = params; p[d] += fd_eps; apply(p);
            double L1 = loss();
            g[d] = (L1 - L0)/fd_eps;
            apply(params); // restore
        }
    };
    for(int t=1; t<=iters; ++t){
        std::vector<double> g; grad(g);
        for(int d=0; d<D; ++d){
            m[d] = beta1*m[d] + (1.0-beta1)*g[d];
            v[d] = beta2*v[d] + (1.0-beta2)*g[d]*g[d];
            double mhat = m[d]/(1.0-std::pow(beta1,t));
            double vhat = v[d]/(1.0-std::pow(beta2,t));
            params[d] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
        apply(params);
    }
}

} // namespace
