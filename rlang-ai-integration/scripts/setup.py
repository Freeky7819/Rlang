#!/usr/bin/env python3
"""
Setup and Installation Script

Checks dependencies, downloads models, runs tests.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run shell command with nice output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✓ Complete: {description}")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     RLang AI Integration - Setup & Installation         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Install requirements
    if not run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    ):
        sys.exit(1)
    
    # Check PyTorch GPU
    print("\nChecking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ No GPU detected - will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed properly")
        sys.exit(1)
    
    # Download model (optional, will happen on first run anyway)
    print("\nModel will be downloaded on first run (~500MB for GPT-2)")
    
    # Run tests
    if not run_command(
        "python -m pytest tests/ -v",
        "Running unit tests"
    ):
        print("⚠ Some tests failed (may be OK for first setup)")
    
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║                  ✓ Setup Complete!                      ║
║                                                          ║
║  Next steps:                                            ║
║    1. Run pilot test: python scripts/quick_pilot.py    ║
║    2. Review results                                    ║
║    3. Run full experiment if promising                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    main()
