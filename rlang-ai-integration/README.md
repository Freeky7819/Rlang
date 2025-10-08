# RLang AI Integration - Experimental Package

**Version:** 0.1.0  
**Author:** Damjan Žakelj  
**Hardware Requirements:** GPU with 12GB+ VRAM (tested on RTX 5070)

## 🎯 Purpose

Test whether **RLang resonance mechanism** improves AI persona stability across sessions, measured by phase coherence in embedding dynamics.

**Core Hypothesis:** Resonance-based correction at specific frequency omega prevents agent drift better than baseline approaches.

---

## 📦 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pilot Test (5 minutes)

```bash
python scripts/quick_pilot.py
```

This will:
- Load small model (GPT-2 Small, ~500MB)
- Run 3 sessions with baseline + RLang
- Show preliminary results
- Verify everything works

### 3. Full Experiment (2-4 hours)

```bash
python scripts/run_experiment.py --config configs/default.yaml
```

---

## 🏗️ Project Structure

```
rlang-ai-integration/
├── src/
│   ├── models/
│   │   ├── rlang_layer.py         # Core RLang resonance implementation
│   │   ├── augmented_llm.py       # LLM wrapper with RLang
│   │   └── baselines.py           # Control conditions
│   ├── metrics/
│   │   ├── phase_coherence.py     # Primary metric (embedding coherence)
│   │   ├── persona_stability.py   # Secondary metrics
│   │   └── visualization.py       # Plotting utilities
│   ├── experiments/
│   │   ├── runner.py              # Experimental pipeline
│   │   ├── config.py              # Configuration management
│   │   └── analysis.py            # Statistical analysis
│   └── data/
│       ├── questions.json         # Test questions
│       └── personas.json          # Persona templates
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter analysis notebooks
├── configs/                       # Experiment configurations
├── scripts/                       # Utility scripts
├── results/                       # Output directory (created on run)
└── README.md                      # This file
```

---

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  name: "gpt2"              # Options: gpt2, gpt2-medium, microsoft/phi-2
  device: "cuda"            # Auto-detects GPU

rlang:
  omega: 0.9                # Resonance frequency (KEY PARAMETER)
  alpha: 0.12               # Correction amplitude
  use_log_time: true        # Use log(t) vs linear t

experiment:
  n_sessions: 20            # Number of test sessions per condition
  turns_per_session: 30     # Conversation length
  calibration_sessions: 5   # Sessions for anchor calibration

conditions:
  - baseline                # No RLang
  - rlang_theory           # RLang with omega=0.9
  - rlang_wrong_freq       # RLang with omega=1.8 (should fail!)
  - noise_control          # Random noise (same amplitude)
```

---

## 📊 Understanding Results

After experiment completes, check `results/`:

```
results/
├── metrics/
│   ├── phase_coherence_summary.csv    # Primary metric
│   ├── persona_stability.csv          # Secondary metrics
│   └── condition_comparison.json      # Statistical tests
├── plots/
│   ├── coherence_comparison.png       # Main result visualization
│   ├── frequency_spectrum.png         # Spectral analysis
│   └── embedding_trajectories.png     # Embedding dynamics
└── analysis/
    ├── statistical_report.txt         # Detailed statistics
    └── interpretation.md              # What do results mean?
```

**Key Question:** Is `rlang_theory` coherence significantly higher than `baseline`?

---

## 🧪 Experimental Conditions

| Condition | Description | Expected Outcome |
|-----------|-------------|------------------|
| `baseline` | Standard model, no RLang | Baseline drift |
| `rlang_theory` | RLang with omega=0.9 | **Lower drift** (hypothesis) |
| `rlang_wrong_freq` | RLang with omega=1.8 | Same as baseline (test specificity) |
| `noise_control` | Random noise injection | Same as baseline (not just regularization) |

**Critical Test:** If `rlang_theory` ≈ `rlang_wrong_freq`, frequency doesn't matter → theory falsified.

---

## 📈 Metrics Explained

### Primary: Phase Coherence
Measures how consistently embedding dynamics oscillate at target frequency across independent sessions.

**High coherence** = stable persona  
**Low coherence** = drift

### Secondary: Persona Stability
- Response consistency (semantic similarity)
- Value alignment (consistent answers to value-laden questions)
- Style consistency (sentence length, formality, vocabulary)

---

## 🔧 Troubleshooting

### GPU Out of Memory
```bash
# Use smaller model
python scripts/run_experiment.py --model gpt2
```

### Slow Performance
```bash
# Reduce sessions
python scripts/run_experiment.py --n_sessions 10
```

### Installation Issues
```bash
# Install PyTorch with CUDA manually first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📝 Next Steps After Results

### If Positive Results (BF > 5):
1. Run replication with larger model
2. Test on different tasks/personas
3. Begin writing paper
4. Start other domain tests (I1, N1)

### If Negative Results (BF < 2):
1. Publish null result (important!)
2. Analyze failure modes
3. Revise theory
4. Try different omega values (exploratory)

### If Ambiguous (2 < BF < 5):
1. Increase sample size
2. Refine metrics
3. Add ablations
4. Consider boundary conditions

---

## 🤝 Contributing

This is experimental research code. Feedback welcome!

**Contact:** zakelj.damjan@gmail.com

---

## 📄 License

RHL-1.0 (Resonance Harmonic License) - See LICENSE file

---

## 🙏 Acknowledgments

Built on RLang framework by Damjan Žakelj (2025)

Core idea: *"Every coupled process can be treated as a chord of information."*
