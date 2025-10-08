from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, List

def cohen_d_ind(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2)
    if sp2 <= 0:
        return 0.0
    return (x.mean() - y.mean()) / np.sqrt(sp2 + 1e-12)

def bic_gaussian_SSE(x: np.ndarray, k_params: int) -> float:
    """Compute BIC for a univariate Gaussian fit using SSE/n as MLE variance.
    k_params counts free mean/variance parameters.
    """
    n = len(x)
    if n <= 1:
        return np.inf
    sse = np.sum((x - x.mean())**2)
    sigma2 = sse / max(n,1)
    if sigma2 <= 0:
        sigma2 = 1e-12
    return n * np.log(sigma2) + k_params * np.log(max(n,1))

def bf_bic_approx(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate BF_10 via BIC difference: BF_10 ≈ exp((BIC0 - BIC1)/2).
    H0: common mean (k=2: mean+variance)
    H1: separate means, shared variance (k=3)
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    all_xy = np.concatenate([x, y], axis=0)
    bic0 = bic_gaussian_SSE(all_xy, k_params=2)
    # For H1 with shared variance, approximate by pooling residuals around group means
    sse1 = np.sum((x - x.mean())**2) + np.sum((y - y.mean())**2)
    n = len(all_xy)
    sigma2_1 = sse1 / max(n,1)
    bic1 = n * np.log(max(sigma2_1,1e-12)) + 3 * np.log(max(n,1))
    return float(np.exp((bic0 - bic1)/2.0))

def summarize_and_test(blocks: List[Dict]) -> str:
    """Aggregate metrics across blocks and run stats vs baseline.
    Expects each block to be: { 'tag': str, 'metrics': [ { 'r_len': .., 'circ_var': .., 'phase_std': .. }, ... ] }
    """
    # Collect
    metrics_by_tag = {}
    for b in blocks:
        tag = b.get("tag","unknown")
        arr = [m.get("r_len", np.nan) for m in b.get("metrics", [])]
        metrics_by_tag.setdefault(tag, []).extend(arr)

    def fmt_row(tag, arr):
        a = np.asarray(arr, float)
        return f"{tag:>16s}  n={len(a):3d}  mean={a.mean():.4f}  sd={a.std(ddof=1):.4f}"

    tags = list(metrics_by_tag.keys())
    report = []
    report.append("RESonance Pilot — Phase Coherence (r_len) Summary\n")
    for t in tags:
        report.append(fmt_row(t, metrics_by_tag[t]))
    report.append("\nComparisons vs 'baseline' (Welch's t, Cohen d, BF_BIC approx)\n")
    base = np.asarray(metrics_by_tag.get("baseline", []), float)
    for t in tags:
        if t == "baseline":
            continue
        x = base; y = np.asarray(metrics_by_tag[t], float)
        if len(x) > 1 and len(y) > 1:
            tt = stats.ttest_ind(x, y, equal_var=False)
            d = cohen_d_ind(y, x)
            bf = bf_bic_approx(y, x)
            report.append(f"{t:>16s}  t={tt.statistic:.3f}  p={tt.pvalue:.4g}  d={d:.3f}  BF_BIC≈{bf:.2f}")
        else:
            report.append(f"{t:>16s}  (insufficient data for stats)")
    report.append("\nRule-of-thumb sample size for d≈0.5 to reach 80% power (two-sample): n≈64 per group (\u2248 16/d^2).\n")
    return "\n".join(report)
