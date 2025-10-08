from __future__ import annotations
import argparse, json, time
from pathlib import Path
from utils.seed import set_seeds
from src.models.augmented_llm import AugmentedLLM
from analysis.stats import summarize_and_test

def run_block(tag: str, model_name: str, omega, alpha: float, n_sessions: int, T: int, base_seed: int):
    metrics = []
    for s in range(n_sessions):
        seed = base_seed + s
        llm = AugmentedLLM(model_name=model_name, omega=omega, alpha=alpha, T=T, simulate=True, rng_seed=seed)
        m = llm.run_session_and_measure()
        metrics.append(m)
    return {"tag": tag, "omega": omega, "alpha": alpha, "metrics": metrics}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="sim")  # replace with your real model id if needed
    ap.add_argument("--n_sessions", type=int, default=30)
    ap.add_argument("--T", type=int, default=256, help="timesteps per session")
    ap.add_argument("--omega_list", nargs="+", type=float, default=[0.9, 1.8])
    ap.add_argument("--alpha", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seeds(args.seed)
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path("results")/"runs"/ts
    outdir.mkdir(parents=True, exist_ok=True)

    blocks = []
    blocks.append(run_block("baseline", args.model_name, omega=None, alpha=0.0,
                            n_sessions=args.n_sessions, T=args.T, base_seed=args.seed))
    for om in args.omega_list:
        blocks.append(run_block(f"rlang_ω{om}", args.model_name, omega=om, alpha=args.alpha,
                                n_sessions=args.n_sessions, T=args.T, base_seed=args.seed+1000))
    blocks.append(run_block("noise_control", args.model_name, omega="noise", alpha=args.alpha,
                            n_sessions=args.n_sessions, T=args.T, base_seed=args.seed+2000))

    (outdir/"raw.json").write_text(json.dumps(blocks, indent=2))

    # Flatten per-block metrics to r_len arrays for summary
    # But summarize_and_test expects blocks with 'metrics' list of dicts per session -> already OK
    report = summarize_and_test(blocks)
    (outdir/"statistical_report.txt").write_text(report)
    print(report)
    print(f"\nSaved run to: {outdir}")
