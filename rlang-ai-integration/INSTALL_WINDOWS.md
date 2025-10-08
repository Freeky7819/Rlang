# Installation Instructions (Windows)

## System Requirements

- **GPU:** RTX 5070 (12GB VRAM) or similar
- **RAM:** 32GB recommended
- **CPU:** AMD Ryzen 7 5700X or better
- **OS:** Windows 10/11
- **Python:** 3.8 or higher
- **Disk Space:** ~5GB (model + dependencies)

---

## Step-by-Step Installation

### 1. Extract Archive

```bash
# Extract rlang-ai-integration.zip to desired location
# e.g., C:\Projects\rlang-ai-integration\
```

### 2. Open Command Prompt / PowerShell

```bash
cd C:\Projects\rlang-ai-integration
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

**Command Prompt:**
```bash
venv\Scripts\activate.bat
```

**PowerShell:**
```bash
venv\Scripts\Activate.ps1
```

*Note: If PowerShell gives execution policy error:*
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 5. Install PyTorch with CUDA

**For CUDA 12.1 (RTX 5070):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (older drivers):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6. Install Other Dependencies

```bash
pip install -r requirements.txt
```

*This will take 5-10 minutes.*

### 7. Verify Installation

```bash
python scripts/setup.py
```

This will:
- Check GPU detection
- Download GPT-2 model (~500MB)
- Run basic tests

---

## Quick Test (5 minutes)

```bash
python scripts/quick_pilot.py
```

This runs minimal experiment to verify everything works.

**Expected output:**
```
🚀 RLang AI Integration - Quick Pilot Test
✓ GPU detected: NVIDIA GeForce RTX 5070
  VRAM: 12.0 GB

Loading models...
Running sessions...
Computing coherence...

📊 RESULTS
...
```

---

## Common Issues

### Issue: GPU Not Detected

**Solution:**
1. Check NVIDIA drivers updated: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Restart computer

### Issue: Out of Memory

**Solution:**
Use smaller model in `configs/default.yaml`:
```yaml
model:
  name: "gpt2"  # Instead of gpt2-medium
```

### Issue: Slow Performance

**Solution:**
- Close other GPU applications
- Reduce batch size in config
- Use fewer sessions for testing

### Issue: Import Errors

**Solution:**
```bash
pip install --upgrade transformers accelerate
```

---

## Next Steps

After successful installation:

1. **Run pilot test** (5 min): `python scripts/quick_pilot.py`
2. **Review pilot results** - check if RLang shows improvement
3. **Run full experiment** (2-4 hours): `python scripts/run_experiment.py`
4. **Analyze results** - see `results/` folder

---

## Directory Structure

```
rlang-ai-integration/
├── venv/                  # Virtual environment (created by you)
├── src/                   # Source code
│   ├── models/           # RLang layer, LLM wrapper
│   └── metrics/          # Phase coherence computation
├── scripts/              # Utility scripts
│   ├── setup.py         # Installation check
│   └── quick_pilot.py   # Quick test
├── tests/                # Unit tests
├── configs/              # Configuration files
├── requirements.txt      # Python dependencies
└── README.md            # Main documentation
```

---

## Getting Help

- Check README.md for detailed documentation
- Review test output for error messages
- Ensure GPU drivers are up to date

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Delete entire folder
cd ..
rmdir /s rlang-ai-integration
```

---

**Contact:** zakelj.damjan@gmail.com
