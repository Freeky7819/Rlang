"""
Phase Coherence Metrics

Primary metric for measuring embedding stability across sessions.
Implements cross-spectral density and coherence analysis.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional
import warnings


def compute_phase_coherence(
    embeddings_session1: np.ndarray,
    embeddings_session2: np.ndarray,
    target_omega: float = 0.9,
    fs: float = 1.0,
    n_components: int = 50,
) -> Dict[str, float]:
    """
    Compute phase coherence between two embedding sequences.
    
    This is the PRIMARY METRIC for the experiment. Measures how consistently
    embedding dynamics oscillate at the target resonance frequency across
    independent sessions.
    
    Args:
        embeddings_session1: Embeddings from session 1 [seq_len, hidden_dim]
        embeddings_session2: Embeddings from session 2 [seq_len, hidden_dim]
        target_omega: Target resonance frequency to measure
        fs: Sampling frequency (default 1.0 = 1 sample per turn)
        n_components: Number of PCA components to use
        
    Returns:
        Dictionary with coherence metrics:
            - coherence_at_omega: Coherence at target frequency (KEY METRIC)
            - coherence_broadband: Average coherence across all frequencies
            - phase_lock_index: Alternative phase locking measure
            - spectral_peak_freq: Frequency of maximum coherence
            - snr_at_omega: Signal-to-noise ratio at target frequency
    """
    # Ensure inputs are numpy arrays
    emb1 = np.array(embeddings_session1)
    emb2 = np.array(embeddings_session2)
    
    # Check shapes
    if emb1.shape != emb2.shape:
        raise ValueError(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")
    
    seq_len, hidden_dim = emb1.shape
    
    if seq_len < 10:
        warnings.warn(f"Sequence too short ({seq_len}), coherence may be unreliable")
    
    # Dimensionality reduction via PCA (for computational efficiency)
    n_components = min(n_components, hidden_dim, seq_len)
    
    pca = PCA(n_components=n_components)
    traj1 = pca.fit_transform(emb1)  # [seq_len, n_components]
    traj2 = pca.transform(emb2)       # [seq_len, n_components]
    
    # Compute power spectral density (PSD)
    freqs, psd1 = signal.welch(traj1, fs=fs, axis=0, nperseg=min(seq_len, 16))
    _, psd2 = signal.welch(traj2, fs=fs, axis=0, nperseg=min(seq_len, 16))
    
    # Cross-spectral density (CSD)
    _, csd = signal.csd(traj1, traj2, fs=fs, axis=0, nperseg=min(seq_len, 16))
    
    # Magnitude-squared coherence
    # C(f) = |CSD(f)|^2 / (PSD1(f) * PSD2(f))
    coherence = np.abs(csd)**2 / (psd1 * psd2 + 1e-10)  # [n_freqs, n_components]
    
    # Average across components
    coherence_avg = coherence.mean(axis=1)  # [n_freqs]
    
    # Convert omega to frequency in Hz
    # omega is angular frequency: omega = 2*pi*f
    target_freq_hz = target_omega / (2 * np.pi)
    
    # Find closest frequency bin
    omega_idx = np.argmin(np.abs(freqs - target_freq_hz))
    
    # Extract coherence at target omega
    coherence_at_omega = coherence_avg[omega_idx]
    
    # Broadband coherence (average across all frequencies)
    coherence_broadband = coherence_avg.mean()
    
    # Signal-to-noise ratio at omega (compared to nearby frequencies)
    # Use 3-bin window around omega
    window_start = max(0, omega_idx - 1)
    window_end = min(len(freqs), omega_idx + 2)
    background_indices = list(range(window_start)) + list(range(window_end, len(freqs)))
    
    if len(background_indices) > 0:
        background_coherence = coherence_avg[background_indices].mean()
        snr_at_omega = coherence_at_omega / (background_coherence + 1e-10)
    else:
        snr_at_omega = 1.0
    
    # Find frequency with maximum coherence
    peak_idx = np.argmax(coherence_avg)
    spectral_peak_freq = freqs[peak_idx]
    
    # Phase locking index (alternative metric using Hilbert transform)
    phase_lock_index = compute_phase_locking_index(traj1, traj2)
    
    return {
        'coherence_at_omega': float(coherence_at_omega),
        'coherence_broadband': float(coherence_broadband),
        'phase_lock_index': float(phase_lock_index),
        'spectral_peak_freq': float(spectral_peak_freq),
        'snr_at_omega': float(snr_at_omega),
        'target_omega': target_omega,
        'target_freq_hz': target_freq_hz,
    }


def compute_phase_locking_index(
    traj1: np.ndarray,
    traj2: np.ndarray,
    n_dims: int = 10,
) -> float:
    """
    Compute phase locking index using Hilbert transform.
    
    Alternative to coherence - measures consistency of phase difference
    between two signals.
    
    Args:
        traj1: First trajectory [seq_len, n_components]
        traj2: Second trajectory [seq_len, n_components]
        n_dims: Number of dimensions to average over
        
    Returns:
        Phase locking index [0, 1], where 1 = perfect locking
    """
    from scipy.signal import hilbert
    
    n_dims = min(n_dims, traj1.shape[1])
    phase_locks = []
    
    for dim in range(n_dims):
        # Analytic signal via Hilbert transform
        analytic1 = hilbert(traj1[:, dim])
        analytic2 = hilbert(traj2[:, dim])
        
        # Extract phases
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Phase difference
        phase_diff = phase1 - phase2
        
        # Phase locking value (PLV)
        # PLV = |<exp(i * phase_diff)>|
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        phase_locks.append(plv)
    
    return float(np.mean(phase_locks))


def compute_temporal_coherence_matrix(
    embeddings_list: list,
    target_omega: float = 0.9,
) -> np.ndarray:
    """
    Compute pairwise coherence matrix for multiple sessions.
    
    Useful for visualizing overall stability across many sessions.
    
    Args:
        embeddings_list: List of embedding arrays, one per session
        target_omega: Target resonance frequency
        
    Returns:
        Coherence matrix [n_sessions, n_sessions]
    """
    n_sessions = len(embeddings_list)
    coherence_matrix = np.zeros((n_sessions, n_sessions))
    
    for i in range(n_sessions):
        for j in range(i, n_sessions):
            if i == j:
                coherence_matrix[i, j] = 1.0
            else:
                result = compute_phase_coherence(
                    embeddings_list[i],
                    embeddings_list[j],
                    target_omega=target_omega,
                )
                coh = result['coherence_at_omega']
                coherence_matrix[i, j] = coh
                coherence_matrix[j, i] = coh
    
    return coherence_matrix


def compute_spectral_profile(
    embeddings: np.ndarray,
    fs: float = 1.0,
    n_components: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full spectral profile of embedding dynamics.
    
    Useful for exploratory analysis - shows which frequencies are present.
    
    Args:
        embeddings: Embedding sequence [seq_len, hidden_dim]
        fs: Sampling frequency
        n_components: Number of PCA components
        
    Returns:
        freqs: Frequency array [n_freqs]
        power: Power spectral density [n_freqs]
    """
    seq_len, hidden_dim = embeddings.shape
    n_components = min(n_components, hidden_dim, seq_len)
    
    # PCA reduction
    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(embeddings)
    
    # Compute PSD
    freqs, psd = signal.welch(traj, fs=fs, axis=0, nperseg=min(seq_len, 16))
    
    # Average across components
    power = psd.mean(axis=1)
    
    return freqs, power


if __name__ == '__main__':
    print("Testing phase coherence metrics...")
    
    # Generate synthetic data with known frequency
    t = np.linspace(0, 10, 100)
    omega_true = 0.9
    
    # Two signals with same frequency but noise
    np.random.seed(42)
    signal1 = np.sin(omega_true * t[:, None]) + 0.1 * np.random.randn(100, 64)
    signal2 = np.sin(omega_true * t[:, None] + 0.1) + 0.1 * np.random.randn(100, 64)
    
    # Compute coherence
    result = compute_phase_coherence(signal1, signal2, target_omega=omega_true)
    
    print(f"Coherence at omega={omega_true}: {result['coherence_at_omega']:.4f}")
    print(f"Phase locking index: {result['phase_lock_index']:.4f}")
    print(f"SNR at omega: {result['snr_at_omega']:.4f}")
    print(f"Spectral peak frequency: {result['spectral_peak_freq']:.4f}")
    
    # Test with random signals (should have low coherence)
    random1 = np.random.randn(100, 64)
    random2 = np.random.randn(100, 64)
    
    result_random = compute_phase_coherence(random1, random2, target_omega=omega_true)
    print(f"\nRandom signals coherence: {result_random['coherence_at_omega']:.4f}")
    
    assert result['coherence_at_omega'] > result_random['coherence_at_omega'], \
        "Coherent signals should have higher coherence!"
    
    print("\n✓ All tests passed!")
