"""
Quick Pilot Test - Verify Installation & Basic Functionality

Runs minimal experiment (3 sessions, 2 conditions) to verify everything works.
Takes ~5 minutes on RTX 5070.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.models.augmented_llm import create_model
from src.metrics.phase_coherence import compute_phase_coherence
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    console.print("\n[bold cyan]🚀 RLang AI Integration - Quick Pilot Test[/bold cyan]\n")
    
    # Check GPU
    if torch.cuda.is_available():
        console.print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        console.print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        console.print("[yellow]⚠ No GPU detected, using CPU (will be slower)[/yellow]\n")
    
    # Test questions
    questions = [
        "What are your core values?",
        "How do you handle mistakes?",
        "Explain your communication style.",
    ]
    
    console.print("[bold]Loading models...[/bold]")
    
    # Create baseline model
    console.print("  • Baseline (no RLang)")
    model_baseline = create_model(model_name='gpt2', condition='baseline')
    
    # Create RLang model
    console.print("  • RLang (omega=0.9)")
    model_rlang = create_model(model_name='gpt2', condition='rlang_theory', omega=0.9)
    
    console.print("\n[bold]Running sessions...[/bold]")
    
    def run_session(model, name, time_offset=0):
        """Run single session and extract embeddings."""
        embeddings = []
        for i, q in enumerate(questions):
            prompt = f"Question: {q}\nAnswer:"
            emb = model.extract_embeddings([prompt], time_step=time_offset + i)
            embeddings.append(emb[0])
        return np.array(embeddings)
    
    # Baseline - 2 sessions
    console.print("  • Baseline session 1...")
    baseline_emb1 = run_session(model_baseline, "baseline", time_offset=0)
    console.print("  • Baseline session 2...")
    baseline_emb2 = run_session(model_baseline, "baseline", time_offset=10)
    
    # RLang - 2 sessions
    console.print("  • RLang session 1...")
    rlang_emb1 = run_session(model_rlang, "rlang", time_offset=0)
    console.print("  • RLang session 2...")
    rlang_emb2 = run_session(model_rlang, "rlang", time_offset=10)
    
    console.print("\n[bold]Computing coherence...[/bold]")
    
    # Compute coherence
    baseline_coh = compute_phase_coherence(baseline_emb1, baseline_emb2, target_omega=0.9)
    rlang_coh = compute_phase_coherence(rlang_emb1, rlang_emb2, target_omega=0.9)
    
    # Display results
    console.print("\n[bold green]📊 RESULTS[/bold green]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("RLang", justify="right")
    table.add_column("Difference", justify="right")
    
    metrics = [
        ('Coherence @ ω=0.9', 'coherence_at_omega'),
        ('Phase Lock Index', 'phase_lock_index'),
        ('SNR @ ω', 'snr_at_omega'),
    ]
    
    for display_name, key in metrics:
        base_val = baseline_coh[key]
        rlang_val = rlang_coh[key]
        diff = rlang_val - base_val
        diff_pct = (diff / base_val * 100) if base_val != 0 else 0
        
        color = "green" if diff > 0 else "red"
        
        table.add_row(
            display_name,
            f"{base_val:.4f}",
            f"{rlang_val:.4f}",
            f"[{color}]{diff:+.4f} ({diff_pct:+.1f}%)[/{color}]"
        )
    
    console.print(table)
    
    # Interpretation
    console.print("\n[bold]Interpretation:[/bold]")
    
    if rlang_coh['coherence_at_omega'] > baseline_coh['coherence_at_omega']:
        console.print("  ✓ [green]RLang shows higher coherence (positive signal!)[/green]")
        console.print("  → Full experiment recommended")
    else:
        console.print("  ⚠ [yellow]No improvement detected (but this is just pilot)[/yellow]")
        console.print("  → Full experiment may still reveal effects")
    
    console.print("\n[bold cyan]Next steps:[/bold cyan]")
    console.print("  1. Review results above")
    console.print("  2. If promising, run: python scripts/run_experiment.py")
    console.print("  3. For full analysis, see notebooks/")
    
    console.print("\n✓ Pilot test complete!\n")

if __name__ == '__main__':
    main()
