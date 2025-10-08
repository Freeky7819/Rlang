from __future__ import annotations
import numpy as np

def _as_angle(phases: np.ndarray) -> np.ndarray:
    # Map to principal value [-pi, pi)
    return ( (phases + np.pi) % (2*np.pi) ) - np.pi

def resultant_vector_length(phases: np.ndarray) -> float:
    """Resultant vector length R in [0,1]; 1 = perfectly locked."""
    z = np.exp(1j * _as_angle(phases))
    return np.abs(np.mean(z))

def circular_variance(phases: np.ndarray) -> float:
    """Circular variance V = 1 - R in [0,1]; 0 = perfectly locked."""
    return 1.0 - resultant_vector_length(phases)

def phase_std(phases: np.ndarray) -> float:
    """Circular standard deviation (Fisher, 1993) in radians."""
    R = max(resultant_vector_length(phases), 1e-12)
    return np.sqrt(-2.0 * np.log(R))

def coherence_suite(phases: np.ndarray) -> dict:
    """Compute a minimal, robust set of phase-coherence metrics for a single session.
    phases: 1D array (time,) of instantaneous phase in radians.
    Returns: dict with keys: r_len, circ_var, phase_std.
    """
    phases = np.asarray(phases).astype(float).ravel()
    return {
        "r_len": float(resultant_vector_length(phases)),
        "circ_var": float(circular_variance(phases)),
        "phase_std": float(phase_std(phases)),
    }
