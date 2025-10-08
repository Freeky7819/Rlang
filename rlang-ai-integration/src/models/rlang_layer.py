"""
RLang Resonance Layer for AI Stability

Core implementation of resonance-based drift correction for LLMs.
Based on RLang framework principle: coupled processes as information chords.

Mathematical Foundation:
    delta = alpha * sin(omega * log(t+1) + phase_offset)
    h'_t = h_t + delta * (anchor - h_projected)

Where:
    h_t: Hidden state at time t
    anchor: Reference state (desired persona)
    h_projected: Projection of h_t onto anchor direction
    omega: Resonance frequency (KEY THEORETICAL PARAMETER)
    alpha: Correction strength
    t: Time step (conversation turn)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class RLangResonanceLayer(nn.Module):
    """
    Applies resonance-based stability correction to LLM hidden states.
    
    This layer implements the core RLang mechanism: oscillatory correction
    that pulls hidden states back toward an anchor (reference persona) with
    phase-modulated strength.
    
    Args:
        hidden_dim: Dimensionality of hidden states
        omega: Resonance frequency (default 0.9 from theory)
        alpha: Correction amplitude (default 0.12)
        use_log_time: Use log(t) instead of linear t (default True)
        learnable_anchor: Whether anchor is learnable parameter (default True)
        per_dim_phase: Use different phase offset per dimension (default True)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        omega: float = 0.9,
        alpha: float = 0.12,
        use_log_time: bool = True,
        learnable_anchor: bool = True,
        per_dim_phase: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.omega = omega
        self.alpha = alpha
        self.use_log_time = use_log_time
        self.per_dim_phase = per_dim_phase
        
        # Initialize anchor vector (reference state)
        anchor_init = torch.randn(hidden_dim) / np.sqrt(hidden_dim)
        if learnable_anchor:
            self.anchor = nn.Parameter(anchor_init)
        else:
            self.register_buffer('anchor', anchor_init)
        
        # Phase offsets (allows richer dynamics across dimensions)
        if per_dim_phase:
            phase_init = torch.rand(hidden_dim) * 2 * np.pi
            self.phase_offsets = nn.Parameter(phase_init)
        else:
            self.register_buffer('phase_offsets', torch.zeros(hidden_dim))
        
        # Track statistics for monitoring
        self.register_buffer('correction_magnitude', torch.tensor(0.0))
        self.register_buffer('n_forward_calls', torch.tensor(0))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        time_step: int,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Apply resonance correction to hidden states.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            time_step: Current conversation turn number (0-indexed)
            return_components: If True, return dict with correction components
            
        Returns:
            corrected_hidden: Corrected hidden states (same shape as input)
            (optional) components: Dict with intermediate values for analysis
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Normalize anchor to unit vector
        anchor_norm = self.anchor / (self.anchor.norm() + 1e-8)
        
        # Project hidden state onto anchor direction
        # projection[b, s] = how much h_t aligns with anchor
        projection = torch.einsum('bsd,d->bs', hidden_states, anchor_norm)
        
        # Reconstruct projection in full space
        h_projected = projection.unsqueeze(-1) * anchor_norm.unsqueeze(0).unsqueeze(0)
        
        # Deviation from anchor (what we want to correct)
        deviation = hidden_states - h_projected
        
        # Compute time signal
        if self.use_log_time:
            time_signal = np.log(time_step + 1.0)  # log(1) = 0 at t=0
        else:
            time_signal = float(time_step)
        
        # Phase-modulated oscillation: sin(omega * t + phi)
        phases = self.omega * time_signal + self.phase_offsets
        modulation = torch.sin(phases)  # [hidden_dim]
        
        # Scale by amplitude
        modulation = self.alpha * modulation
        
        # Apply correction: pull back toward anchor
        # Positive modulation → pull toward anchor
        # Negative modulation → allow drift (but will reverse later)
        correction = modulation.view(1, 1, -1) * deviation
        
        # Final corrected state
        corrected_hidden = hidden_states + correction
        
        # Update statistics (for monitoring)
        with torch.no_grad():
            self.correction_magnitude = correction.abs().mean()
            self.n_forward_calls += 1
        
        if return_components:
            components = {
                'projection': projection,
                'h_projected': h_projected,
                'deviation': deviation,
                'modulation': modulation,
                'correction': correction,
                'anchor_norm': anchor_norm,
            }
            return corrected_hidden, components
        
        return corrected_hidden
    
    def set_anchor(self, anchor: torch.Tensor):
        """
        Set anchor from calibration phase.
        
        Args:
            anchor: Pre-computed anchor vector [hidden_dim]
        """
        with torch.no_grad():
            anchor_normalized = anchor / (anchor.norm() + 1e-8)
            self.anchor.copy_(anchor_normalized)
    
    def get_stats(self) -> Dict[str, float]:
        """Return current statistics for monitoring."""
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'correction_magnitude': self.correction_magnitude.item(),
            'n_forward_calls': self.n_forward_calls.item(),
            'anchor_norm': self.anchor.norm().item(),
        }
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'hidden_dim={self.hidden_dim}, omega={self.omega:.3f}, '
                f'alpha={self.alpha:.3f}, use_log_time={self.use_log_time}')


class NoiseControlLayer(nn.Module):
    """
    Control condition: Random noise injection (same amplitude as RLang).
    
    This tests whether RLang benefits are due to resonance specifically,
    or just from adding any regularizing noise.
    """
    
    def __init__(self, hidden_dim: int, noise_std: float = 0.12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        
    def forward(self, hidden_states: torch.Tensor, time_step: int) -> torch.Tensor:
        noise = torch.randn_like(hidden_states) * self.noise_std
        return hidden_states + noise


class SquareWaveLayer(nn.Module):
    """
    Control condition: Square wave modulation instead of sinusoidal.
    
    Tests whether smooth sinusoidal dynamics are necessary,
    or any periodic signal works.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        omega: float = 0.9,
        alpha: float = 0.12,
        use_log_time: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.omega = omega
        self.alpha = alpha
        self.use_log_time = use_log_time
        
        anchor_init = torch.randn(hidden_dim) / np.sqrt(hidden_dim)
        self.register_buffer('anchor', anchor_init)
        
    def forward(self, hidden_states: torch.Tensor, time_step: int) -> torch.Tensor:
        anchor_norm = self.anchor / (self.anchor.norm() + 1e-8)
        
        projection = torch.einsum('bsd,d->bs', hidden_states, anchor_norm)
        h_projected = projection.unsqueeze(-1) * anchor_norm.unsqueeze(0).unsqueeze(0)
        deviation = hidden_states - h_projected
        
        time_signal = np.log(time_step + 1.0) if self.use_log_time else float(time_step)
        
        # Square wave: +1 or -1
        phase = self.omega * time_signal
        square_wave = torch.sign(torch.sin(torch.tensor(phase)))
        
        correction = self.alpha * square_wave * deviation
        return hidden_states + correction


# Factory function for easy condition creation
def create_correction_layer(
    condition: str,
    hidden_dim: int,
    omega: float = 0.9,
    alpha: float = 0.12,
    **kwargs
) -> nn.Module:
    """
    Factory to create correction layer based on experimental condition.
    
    Args:
        condition: One of ['rlang', 'noise', 'square_wave', 'none']
        hidden_dim: Hidden state dimensionality
        omega: Resonance frequency (for rlang, square_wave)
        alpha: Correction amplitude
        **kwargs: Additional arguments passed to layer constructor
        
    Returns:
        Correction layer module
    """
    if condition == 'rlang':
        return RLangResonanceLayer(hidden_dim, omega, alpha, **kwargs)
    elif condition == 'noise':
        return NoiseControlLayer(hidden_dim, noise_std=alpha)
    elif condition == 'square_wave':
        return SquareWaveLayer(hidden_dim, omega, alpha, **kwargs)
    elif condition == 'none':
        return nn.Identity()  # No correction
    else:
        raise ValueError(f"Unknown condition: {condition}")


if __name__ == '__main__':
    # Quick test
    print("Testing RLangResonanceLayer...")
    
    layer = RLangResonanceLayer(hidden_dim=768, omega=0.9, alpha=0.12)
    hidden = torch.randn(2, 10, 768)  # [batch=2, seq=10, hidden=768]
    
    # Test forward pass
    output = layer(hidden, time_step=5)
    assert output.shape == hidden.shape, "Shape mismatch!"
    
    # Test with components
    output, components = layer(hidden, time_step=5, return_components=True)
    assert 'correction' in components, "Components missing!"
    
    # Test statistics
    stats = layer.get_stats()
    print(f"Stats: {stats}")
    
    # Test anchor setting
    new_anchor = torch.randn(768)
    layer.set_anchor(new_anchor)
    
    print("✓ All tests passed!")
    print(f"Layer: {layer}")
