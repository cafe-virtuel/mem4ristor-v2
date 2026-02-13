import sys
import os
import numpy as np
import pytest
import random
import yaml
import warnings

# Ignorer les warnings de division par zéro car on les cherche
warnings.filterwarnings("ignore", category=RuntimeWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mem4ristor.core import Mem4ristorV2

def test_extreme_initialization_chaos():
    """Tente d'initialiser le modèle avec des tailles absurdes."""
    print("\n[CHAOS] Testing extreme N values...")
    for n in [0, -1, 10**9, "100", 1.5]:
        try:
            model = Mem4ristorV2()
            model._initialize_params(N=n)
            print(f"  (!) Warning: Model accepted N={n}")
        except (ValueError, TypeError, MemoryError) as e:
            print(f"  [OK] Rejected N={n}: {e}")

def test_config_injection_attack():
    """Injecte des types de données malveillants dans la configuration."""
    print("\n[CHAOS] Testing config injection...")
    evil_configs = [
        {'dynamics': {'v_cubic_divisor': 0}}, # Division par zéro
        {'dynamics': {'v_cubic_divisor': -1e-20}}, # Diviseur négatif minuscule
        {'dynamics': {'dt': 1e10}}, # Pas de temps géant
        {'dynamics': {'dt': -0.05}}, # Pas de temps négatif
        {'doubt': {'tau_u': 0}}, # Division par zéro dans du/dt
        {'coupling': {'D': np.inf}}, # Couplage infini
        {'noise': {'sigma_v': "0.05"}}, # Type incorrect
        {'noise': {'use_rtn': True, 'rtn_p_flip': 2.0}}, # Probabilité > 1
    ]
    
    for cfg in evil_configs:
        try:
            model = Mem4ristorV2(config=cfg)
            # Tenter un step pour voir si ça explose
            model.step()
            v_val = model.v[0]
            if np.isnan(v_val) or np.isinf(v_val):
                 print(f"  [FAIL] Model produced {v_val} with config {cfg}")
            else:
                 print(f"  [WARNING] Model survived evil config {cfg} with v={v_val}")
        except (ValueError, TypeError, ZeroDivisionError) as e:
            print(f"  [OK] Caught evil config {cfg}: {e}")

def test_stimulus_overflow_and_types():
    """Bombarde le modèle avec des stimuli de toutes sortes."""
    print("\n[CHAOS] Testing stimulus torture...")
    model = Mem4ristorV2()
    model._initialize_params(N=10)
    
    torture_inputs = [
        np.nan,
        np.inf,
        "un million",
        [1, 2], # Mauvaise taille
        np.array([1]*11), # Mauvaise taille
        1e308,
        -1e308,
        {"value": 0.5},
        None
    ]
    
    for inp in torture_inputs:
        try:
            model.step(I_stimulus=inp)
            if np.any(np.isnan(model.v)) or np.any(np.isinf(model.v)):
                print(f"  [FAIL] NaN/Inf detected after stimulus: {inp}")
        except Exception as e:
            print(f"  [OK] Blocked/Handled stimulus {type(inp)}: {e}")

def test_memory_leak_and_performance_hit():
    """Vérifie si le modèle s'alourdit avec le temps ou des tailles croissantes."""
    print("\n[CHAOS] Testing scale and memory...")
    import time
    
    sizes = [10, 100, 1000, 5000]
    for n in sizes:
        start = time.time()
        try:
            model = Mem4ristorV2()
            model._initialize_params(N=n)
            for _ in range(100):
                model.step(I_stimulus=0.5)
            end = time.time()
            print(f"  N={n}: 100 steps in {end-start:.4f}s")
        except MemoryError:
            print(f"  N={n}: Memory Error (Expected for very large N)")
            break

def test_state_corruption_manual():
    """Corrompt manuellement les variables d'état pour voir si les GARDES fonctionnent."""
    print("\n[CHAOS] Testing manual state corruption recovery...")
    model = Mem4ristorV2()
    model._initialize_params(N=10)
    
    # Injection directe de NaN
    model.v[0] = np.nan
    model.w[5] = np.inf
    model.u[2] = -10.0 # Hors bornes [0,1]
    
    # Le prochain step devrait nettoyer/clamer
    model.step()
    
    assert not np.isnan(model.v[0]), "NaN not recovered in v"
    # v2.9.4 Fix: Allow for small noise deviation (sigma_v ~ 0.05)
    # The guard sets v=0.0, but noise is added in the same step
    assert abs(model.v[0]) < 0.5, f"NaN recovered but value {model.v[0]} exploded"
    
    assert not np.isinf(model.w[5]), "Inf not recovered in w"
    assert model.u[2] >= 0.0, "Negative u not clamped"
    print("  [OK] State guards recovered from manual corruption.")

if __name__ == "__main__":
    test_extreme_initialization_chaos()
    test_config_injection_attack()
    test_stimulus_overflow_and_types()
    test_memory_leak_and_performance_hit()
    test_state_corruption_manual()
