# RLang Mathematical Foundation
Date: 2025-10-08

## 1. Core Update Rule
For state `(phase, amp, seed)` and profile `(chord, alpha, omega)`:
Δ_i = α * sin(ω * φ_i + c_i)
φ_i <- φ_i + Δ_i
A_i <- A_i + 0.1 * Δ_i
Seed update uses an LCG with a=6364136223846793005, b=1 (mod 2^64).

Why sin: sine is phase-symmetric with zero mean over full cycles, acting as an energy-neutral phase corrector (no cumulative bias). Cosine is equivalent to a phase-shift.

## 2. Chord as Coupling
chord[i] encodes per-dimension phase offset/coupling (Kuramoto analogy). It determines constructive/destructive interference per component.

## 3. Relation to Kuramoto / Oscillatory Networks
Kuramoto: θ̇_i = ω_i + K Σ_j sin(θ_j − θ_i).
Rlang is a discrete-time single-step with an external chord as structured coupling.
In embeddings, phase-lock ≈ directional locking (high cosine to anchor/goal).

## 4. Embedding Locking (Phase → Value)
Let b_t∈R^d be an embedding, anchor u.
Corrective term:  -α * sin(ω ln(t+1) + φ) * (b_t − proj_u(b_t))
acts as a bounded phase-dependent contraction around anchor subspace → bounded drift with adaptation.

## 5. Convergence (Sketch)
For small α, |sin|≤1 ⇒ corrective magnitude ≤ α||b_t − proj_u(b_t)||.
With contraction to goal + bounded perturbations, the system stays in a bounded oscillatory regime (Floquet-like). A Lyapunov-like function over ||b_t − proj_u(b_t)|| gives a formal path to proof.

## 6. Related Work (Pointers)
- Kuramoto (1975), Strogatz (synchronization)
- Hopf bifurcation; Floquet theory
- Attractor networks (stable patterns via recurrence)
- Phase response curves

## 7. Implications
- AI: reduces long-horizon embedding drift without retraining.
- Control/Signals: stabilizes iterative filters with low latency.
- Creative systems: preserves style under exploration.
