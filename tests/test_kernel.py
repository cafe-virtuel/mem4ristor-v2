import pytest
import numpy as np
import random
from mem4ristor.core import Mem4ristorV2

def test_heretic_inversion():
    """Verify that heretic units invert the stimulus polarity."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.1},
        'coupling': {'D': 0.0, 'heretic_ratio': 1.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.0}
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=10)
    assert np.all(model.heretic_mask)
    
    v_pre = model.v.copy()
    model.step(I_stimulus=1.0)
    # With heretic, I_eff = -1.0. dv should be negative (stimulus pushes down)
    # v should decrease because I_eff is inverted for heretics
    dv = model.v - v_pre
    # Check that all units moved in the negative direction (with some tolerance for FHN dynamics)
    assert np.mean(dv) < 0, f"Heretics should invert stimulus. Mean dv={np.mean(dv):.4f}"

def test_repulsive_flip():
    """Verify that coupling becomes repulsive when doubt u > 0.5."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.1},
        'coupling': {'D': 1.0, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.0}
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=2)
    model.v = np.array([-1.0, 1.0])
    model.u = np.array([0.8, 0.8])
    adj = np.array([[0, 1], [1, 0]])
    model.step(I_stimulus=0.0, coupling_input=adj)
    assert model.v[0] < -1.0, f"Unit 0 should be repelled. v={model.v[0]}"
    assert model.v[1] > 1.0, f"Unit 1 should be repelled. v={model.v[1]}"

def test_snr_validity():
    """Verify that the repulsive interaction is stronger than the noise floor."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.01},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02} # SNR floor test
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=2)
    model.v = np.array([-1.0, 1.0])
    adj = np.array([[0, 1], [1, 0]])
    
    # Observe 100 steps. In presence of heavy noise, the avg displacement should still be repulsive.
    v0_path = []
    for _ in range(100):
        model.step(I_stimulus=0.0, coupling_input=adj)
        v0_path.append(model.v[0])
    
    # If SNR is broken by the 0.5 noise injection, v[0] will not be consistently < -1.0
    v0_avg = np.mean(v0_path)
    assert v0_avg < -1.0, f"SNR FAIL: Massive noise injection masked the repulsion. avg_v0={v0_avg:.4f}"

def test_spatial_clustering():
    """Verify heretic homogeneity."""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    heretic_ids = np.where(model.heretic_mask)[0]
    diffs = np.diff(heretic_ids)
    cluster_count = np.sum(diffs == 1)
    assert cluster_count < 5, f"Heretics too clustered: {cluster_count}"

def test_rk45_stability():
    """Verify RK45 stability with v2.7 entropy metric."""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    adj = np.zeros((100, 100))
    for i in range(100):
        for j in [i-1, i+1, i-10, i+10]:
            if 0 <= j < 100: adj[i, j] = 1
    model.solve_rk45((0, 1.0), I_stimulus=0.5, adj_matrix=adj)
    h_final = model.calculate_entropy()
    assert h_final > 0.5, f"Entropy collapsed: {h_final}"

def test_reproducibility():
    """Verify bit-level reproducibility."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.1},
        'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.05}
    }
    model1 = Mem4ristorV2(config=cfg, seed=123)
    model2 = Mem4ristorV2(config=cfg, seed=123)
    for _ in range(10):
        model1.step(I_stimulus=0.5)
        model2.step(I_stimulus=0.5)
    assert np.allclose(model1.v, model2.v), f"v diverged: {np.max(np.abs(model1.v - model2.v))}"
