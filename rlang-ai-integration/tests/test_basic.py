"""
Basic unit tests for RLang components
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.rlang_layer import RLangResonanceLayer
from src.metrics.phase_coherence import compute_phase_coherence


def test_rlang_layer_forward():
    """Test RLang layer forward pass."""
    layer = RLangResonanceLayer(hidden_dim=128, omega=0.9, alpha=0.12)
    
    # Create dummy input
    hidden = torch.randn(2, 10, 128)  # [batch, seq, hidden]
    
    # Forward pass
    output = layer(hidden, time_step=5)
    
    # Check output shape
    assert output.shape == hidden.shape
    
    # Check output is different from input (correction applied)
    assert not torch.allclose(output, hidden)


def test_rlang_layer_with_components():
    """Test RLang layer returns components."""
    layer = RLangResonanceLayer(hidden_dim=128)
    hidden = torch.randn(2, 10, 128)
    
    output, components = layer(hidden, time_step=5, return_components=True)
    
    # Check all components present
    required_keys = ['projection', 'h_projected', 'deviation', 'modulation', 'correction']
    for key in required_keys:
        assert key in components


def test_phase_coherence_basic():
    """Test phase coherence computation."""
    # Create synthetic signals with known frequency
    t = np.linspace(0, 10, 50)
    omega = 0.9
    
    signal1 = np.sin(omega * t[:, None]) + 0.1 * np.random.randn(50, 32)
    signal2 = np.sin(omega * t[:, None]) + 0.1 * np.random.randn(50, 32)
    
    result = compute_phase_coherence(signal1, signal2, target_omega=omega)
    
    # Check all metrics present
    assert 'coherence_at_omega' in result
    assert 'phase_lock_index' in result
    assert 'snr_at_omega' in result
    
    # Coherence should be between 0 and 1
    assert 0 <= result['coherence_at_omega'] <= 1
    assert 0 <= result['phase_lock_index'] <= 1


def test_phase_coherence_random_signals():
    """Test that random signals have low coherence."""
    random1 = np.random.randn(50, 32)
    random2 = np.random.randn(50, 32)
    
    result = compute_phase_coherence(random1, random2, target_omega=0.9)
    
    # Random signals should have low coherence
    assert result['coherence_at_omega'] < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
