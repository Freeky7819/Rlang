# RLang Add‑ons Pack (v1)

This folder contains the **missing** or **suggested** files to make your pilot reproducible and aligned with the docs:

## What’s inside
- `scripts/run_experiment.py` — one‑command runner for a full baseline vs RLang vs noise_control experiment.
- `configs/default.yaml` — minimal config aligned with docs.
- `src/models/augmented_llm.py` — a **runnable fallback simulator** (replace with your real LLM pipeline later).
- `src/metrics/phase_coherence.py` — operational definitions: resultant vector length (R), circular variance, circular std.
- `analysis/stats.py` — stats summary (Welch t‑test, Cohen’s d, **BF_BIC approx**).
- `utils/seed.py` — strong seeding & deterministic switches (Torch optional).
- `assets/prompts/pilot_en.txt` — placeholder prompts.
- `requirements.txt` — pinned minimal dependencies.

## Quickstart (pilot, no external APIs needed)
```bash
pip install -r requirements.txt
python scripts/run_experiment.py --n_sessions 30 --omega_list 0.9 1.8 --alpha 0.12
```
Outputs are saved under `results/runs/<timestamp>/` including `statistical_report.txt`.

## Notes
- The Bayes factor reported is a **BIC-based approximation** (`BF_BIC≈exp((BIC0-BIC1)/2)`). For rigorous Bayesian t‑tests (JZS), swap in your preferred library later.
- The simulator produces log‑periodic torques on phase velocity to emulate the resonance effect. Use it only for pipeline validation; the real study should plug in actual session embeddings/phases.
