import numpy as np
import pytest
from mem4ristor.core import Mem4ristorV2

@pytest.mark.xfail(reason="Known theoretical limitation: SNR collapse at high noise regimes.")
def test_snr_significance_breakdown():
    """Verify signal integrity against noise floor."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.15} # INTENTIONALLY HIGH NOISE
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=100)
    
    # Calculate effective coupling signal strength
    # Max signal is when u=0 or u=1 -> |(1-2u)| = 1
    # Max I_coup component ~ D_eff * 1.0 * L_ij * (v_j - v_i)
    # L_ij avg is ~1/sqrt(N)
    
    signal_strength = model.D_eff * 1.0 # Rough order of magnitude
    noise_floor = cfg['noise']['sigma_v']
    
    snr = signal_strength / noise_floor
    print(f"Adversarial SNR: {snr:.4f}")
    
    # Requirement: SNR > 3.0 for signal readability
    assert snr > 3.0, f"Signal-to-Noise Ratio too low (SNR={snr:.2f}). Repulsion signal buried in noise."

@pytest.mark.xfail(reason="Known numerical instability at dt > 0.1. Recommended dt <= 0.05.")
def test_euler_drift_torture():
    """Verify Euler integration stability at high time steps."""
    # Stress test for numerical drift
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.5}, # High DT
        'coupling': {'D': 0.5, 'heretic_ratio': 0.5},
        'doubt': {'epsilon_u': 0.1, 'k_u': 2.0, 'sigma_baseline': 0.0, 'u_clamp': [0.0, 1.0], 'tau_u': 10.0},
        'noise': {'sigma_v': 0.01}
    }
    model = Mem4ristorV2(config=cfg, seed=123)
    model._initialize_params(N=625)
    
    h_initial = model.calculate_entropy()
    
    # Run simulation for a large number of steps
    for _ in range(200):
        model.step()
        
    h_final = model.calculate_entropy()
    
    # If entropy collapses at high dt, it proves numerical instability
    assert h_final > 0.5 * h_initial, f"ADVERSARIAL FAIL: Entropy collapse at high dt (dt=0.5). H={h_final:.2f}"

def test_scale_invariance_failure():
    """Test claim: Robust up to N=1000. Try to break it at N=1200."""
    try:
        model = Mem4ristorV2(seed=42)
        model._initialize_params(N=1200)
        # This should test memory limits or execution time
        for _ in range(10):
            model.step()
    except MemoryError:
        pytest.fail("ADVERSARIAL FAIL: Out of memory at N=1200. Claim 'Scale Invariant' is false.")
    except Exception as e:
        pytest.fail(f"ADVERSARIAL FAIL: Unexpected crash at N=1200: {str(e)}")
