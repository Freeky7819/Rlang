# ⚡ QUICK START GUIDE

Get running in 10 minutes!

## 📋 Prerequisites

- Windows 10/11
- RTX 5070 (12GB VRAM)
- Python 3.8+

## 🚀 Installation (5 minutes)

### 1. Extract & Navigate
```bash
# Extract ZIP to C:\Projects\
cd C:\Projects\rlang-ai-integration
```

### 2. Create Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch (with CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify
```bash
python scripts/setup.py
```

**✓ Should see:** "GPU detected: NVIDIA GeForce RTX 5070"

---

## 🧪 Run Pilot Test (5 minutes)

```bash
python scripts/quick_pilot.py
```

**What it does:**
- Loads GPT-2 Small (~500MB, auto-downloads)
- Tests baseline vs RLang (3 questions each)
- Shows coherence comparison

**Expected output:**
```
📊 RESULTS

Metric              Baseline    RLang       Difference
─────────────────────────────────────────────────────
Coherence @ ω=0.9   0.XXXX     0.YYYY      +X.XX%
Phase Lock Index    0.XXXX     0.YYYY      +X.XX%
...
```

**Interpretation:**
- ✅ **Positive difference** = RLang working (proceed to full experiment)
- ⚠️ **Negative/zero** = Need more sessions or parameter tuning

---

## 📊 Full Experiment (2-4 hours)

Only run if pilot shows promise!

```bash
python scripts/run_experiment.py
```

**What it does:**
- Runs 20 sessions per condition (baseline, RLang, controls)
- ~2-4 hours on RTX 5070
- Saves results to `results/`

**Outputs:**
```
results/
├── metrics/
│   └── coherence_summary.csv     ← Main results
├── plots/
│   └── coherence_comparison.png  ← Visualization
└── analysis/
    └── statistical_report.txt    ← Is it significant?
```

---

## 📈 Interpret Results

### Check: `results/analysis/statistical_report.txt`

**Look for:**
```
PRIMARY TEST (RLang vs Baseline):
  Bayes Factor: X.XX
  p-value: 0.XXX
  Cohen's d: X.XX
  
  SUCCESS: [TRUE/FALSE]
```

**Success criteria:**
- ✅ **BF > 5** and **p < 0.01** = Strong evidence for RLang
- ⚠️ **BF 2-5** = Weak evidence, need more data
- ❌ **BF < 2** = No evidence, theory falsified

---

## 🎯 What to Do Next

### If Successful (BF > 5):
1. 🎉 Celebrate - you have evidence!
2. 📝 Document findings
3. 🔁 Replicate with larger model
4. 📧 Contact me: zakelj.damjan@gmail.com

### If Ambiguous (BF 2-5):
1. 🔄 Run more sessions (`--n_sessions 50`)
2. 🎛️ Try different omega values
3. 🔍 Analyze failure modes

### If Negative (BF < 2):
1. 📊 Publish null result (important!)
2. 🤔 Revise theory
3. 🧪 Try other domain tests (I1, N1)

---

## 🆘 Troubleshooting

### GPU not detected?
```bash
# Check drivers
nvidia-smi

# Reinstall PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory?
Edit `configs/default.yaml`:
```yaml
model:
  name: "gpt2"  # Use smaller model
```

### Slow?
```bash
# Reduce sessions
python scripts/run_experiment.py --n_sessions 10
```

---

## 📚 More Information

- **Full docs:** README.md
- **Windows install:** INSTALL_WINDOWS.md  
- **Questions:** zakelj.damjan@gmail.com

---

**Built on RLang by Damjan Žakelj (2025)**

*"Every coupled process can be treated as a chord of information."*
