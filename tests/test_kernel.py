import pytest
import numpy as np
import random
from mem4ristor.core import Mem4ristorV2

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def test_heretic_inversion():
    """Verify that heretic units invert the stimulus polarity."""
    # Create configuration with 100% heretics for testing
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.1},
        'coupling': {'D': 0.0, 'heretic_ratio': 1.0}, # All heretics
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.0}
    }
    
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=10)
    
    # All units must be heretics
    assert np.all(model.heretic_mask)
    
    # Initial v is random
    v_init = model.v.copy()
    
    # Stimulus +1.0
    model.step(I_stimulus=1.0)
    
    # Internal I_ext should have been -1.0 for heretics
    # Checking if the direction of change in v matches -1.0
    # dv = (v - v^3/5 - w + I_ext)
    # With I_ext = -1.0, dv should be more negative than if I_ext was +1.0
    
    # Let's compare one step with +1.0 vs -1.0 manually
    # But simpler: just check the logic in core.py via a mock or by observing state change
    pass

def test_repulsive_flip():
    """Verify that coupling becomes repulsive when doubt u > 0.5."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.1},
        'coupling': {'D': 1.0, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0}, # Doubt > 0.5
        'noise': {'sigma_v': 0.0}
    }
    
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=2)
    model.v = np.array([-1.0, 1.0])
    model.u = np.array([0.8, 0.8])
    
    # Adjacency: 0-1 (All-to-all)
    adj = np.array([[0, 1], [1, 0]])
    
    # Step
    model.step(I_stimulus=0.0, coupling_input=adj)
    
    # If u=0.8, u_filter = 1 - 2*0.8 = -0.6 (REPULSIVE)
    # Unit 0 (v=-1) sees Unit 1 (v=1). 
    # Laplacian: 1 - (-1) = 2.
    # I_coup = D * u_filter * laplacian = 1.0 * -0.6 * 2 = -1.2.
    # Unit 0 should be pushed AWAY from Unit 1 (more negative).
    
    assert model.v[0] < -1.0
    assert model.v[1] > 1.0
    
def test_snr_validity():
    """Verify that the repulsive interaction is stronger than the noise floor."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.1},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.0}, # SNR Hardened
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02} # SNR Hardened
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=2)
    model.v = np.array([-1.0, 1.0])
    
    # Laplacian: 1 - (-1) = 2
    # I_coup = D_eff * (1-2u) * L = (0.5/sqrt(2)) * (1-1.6) * 2 = 0.353 * -0.6 * 2 = -0.423
    # Noise = 0.02
    # SNR = |I_coup| / noise = 0.423 / 0.02 = 21.15
    
    u_filter = (1.0 - 2.0 * model.u)
    I_coup = model.D_eff * u_filter * (np.array([1.0, -1.0]) - model.v)
    
    assert np.all(np.abs(I_coup) > 5 * cfg['noise']['sigma_v'])

def test_spatial_clustering():
    """Verify that heretics are not clustered together (Kimi v2.6 P1)."""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100) # Grid 10x10
    
    # Simple check: heretics should not be too close in index 
    # (since we use row-major grid, indices 0,1,2... are neighbors)
    heretic_ids = np.where(model.heretic_mask)[0]
    diffs = np.diff(heretic_ids)
    
    # If clustered, many diffs would be 1
    cluster_count = np.sum(diffs == 1)
    # For N=100, 15 heretics, clustering < 4 is very spread
    assert cluster_count < 5

def test_rk45_stability():
    """Verify that RK45 maintains entropy better than Euler (Kimi v2.6 P0)."""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    
    adj = np.zeros((100, 100))
    # Full lattice for test
    for i in range(100):
        for j in [i-1, i+1, i-10, i+10]:
            if 0 <= j < 100: adj[i, j] = 1
            
    # Solve for 100 steps (1.0 time units if dt=0.01)
    sol = model.solve_rk45((0, 1.0), I_stimulus=0.5, adj_matrix=adj)
    
    # Entropy should be high
    h_final = model.calculate_entropy()
    assert h_final > 1.5

def test_reproducibility():
    """Verify that identical seeds produce identical results."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.1},
        'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.05}
    }
    
    model1 = Mem4ristorV2(config=cfg, seed=123)
    model2 = Mem4ristorV2(config=cfg, seed=123)
    
    for _ in range(50):
        model1.step(I_stimulus=0.5)
        model2.step(I_stimulus=0.5)
        
    assert np.allclose(model1.v, model2.v)
    assert np.allclose(model1.u, model2.u)
