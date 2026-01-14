"""
KIMI v2.0 — CONTRAINTES INÉDITES
Tests adversariaux jamais employés auparavant.
Ces tests sont conçus pour CASSER le code, pas pour le valider.
"""
import numpy as np
import pytest
from mem4ristor.core import Mem4ristorV2, Mem4Network


# =============================================================================
# ATTAQUE 1: INJECTION DE NaN (Corruption Mémoire)
# =============================================================================
def test_nan_injection_survial():
    """Le modèle doit-il détecter ou propager les NaN?"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    
    # Injection d'un NaN dans v
    model.v[50] = np.nan
    
    # Le step va-t-il propager le NaN à tout le réseau?
    model.step(I_stimulus=0.5)
    
    nan_count = np.isnan(model.v).sum()
    
    # KIMI VERDICT: NaN doit soit être détecté, soit ne pas se propager
    assert nan_count <= 1, f"NaN PROPAGATION: {nan_count}/100 unités contaminées. Le modèle n'a pas de garde-fou."


def test_nan_in_coupling_matrix():
    """NaN dans la matrice de couplage"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=10)
    
    adj = np.ones((10, 10)) * 0.1
    adj[5, 5] = np.nan
    
    try:
        model.step(I_stimulus=0.0, coupling_input=adj)
        v_nan = np.isnan(model.v).any()
        assert not v_nan, "NaN dans adj a contaminé v"
    except (ValueError, FloatingPointError):
        pass  # Acceptable: le modèle refuse les entrées corrompues


# =============================================================================
# ATTAQUE 2: CAS LIMITES (N très petit)
# =============================================================================
def test_single_unit_network():
    """N=1 est-il stable?"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=1)
    
    for _ in range(100):
        model.step(I_stimulus=0.5)
    
    assert np.isfinite(model.v[0]), f"N=1 a divergé: v={model.v[0]}"
    assert np.isfinite(model.u[0]), f"N=1 a divergé: u={model.u[0]}"


def test_two_unit_network_symmetry():
    """N=2 avec conditions initiales symétriques"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=2)
    
    # Forcer symétrie parfaite
    model.v = np.array([0.5, 0.5])
    model.w = np.array([0.5, 0.5])
    model.u = np.array([0.25, 0.25])
    
    # Matrice de couplage symétrique
    adj = np.array([[0, 1], [1, 0]], dtype=float)
    
    # Réinitialiser le RNG pour avoir le même bruit
    model.rng = np.random.RandomState(42)
    
    for _ in range(100):
        model.step(I_stimulus=0.0, coupling_input=adj)
    
    # Les deux unités doivent rester proches (bruit à part)
    diff = np.abs(model.v[0] - model.v[1])
    assert diff < 1.0, f"Brisure de symétrie inattendue: |v0-v1|={diff:.4f}"


# =============================================================================
# ATTAQUE 3: PARAMÈTRES NÉGATIFS OU ABSURDES
# =============================================================================
def test_negative_coupling():
    """D < 0 est-il géré?"""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.05},
        'coupling': {'D': -0.5, 'heretic_ratio': 0.15},  # NÉGATIF
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=100)
    
    # Le modèle devrait soit lever une erreur, soit inverser le comportement
    for _ in range(100):
        model.step()
    
    assert np.all(np.isfinite(model.v)), "D négatif a causé une divergence"


def test_negative_noise():
    """sigma_v < 0 est absurde"""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': -0.1}  # NÉGATIF
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=10)
    
    # numpy.random.normal avec sigma négatif devrait lever une erreur
    with pytest.raises(ValueError):
        model.step()


def test_zero_dt():
    """dt=0 devrait bloquer ou lever une erreur"""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.0},  # ZÉRO
        'coupling': {'D': 0.5, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=10)
    
    v_before = model.v.copy()
    model.step()
    v_after = model.v
    
    # Avec dt=0, rien ne devrait changer (sauf le bruit)
    # En réalité, le bruit est toujours là, donc on vérifie juste la stabilité
    assert np.all(np.isfinite(v_after)), "dt=0 a causé une divergence"


# =============================================================================
# ATTAQUE 4: FRONTIÈRE CRITIQUE u = 0.5
# =============================================================================
def test_critical_doubt_boundary():
    """u = 0.5 exactement: le kernel (1-2u) = 0"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    
    # Forcer u = 0.5 exactement
    model.u[:] = 0.5
    
    adj = np.ones((100, 100)) / 100  # Couplage uniforme
    
    v_before = model.v.copy()
    model.step(I_stimulus=0.0, coupling_input=adj)
    
    # À u=0.5, le terme de couplage est exactement 0
    # Seuls le FHN intrinsèque et le bruit agissent
    assert np.all(np.isfinite(model.v)), "u=0.5 a causé une instabilité"


def test_doubt_at_clamp_boundaries():
    """u à 0.0 et 1.0 exactement"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=10)
    
    model.u[0:5] = 0.0
    model.u[5:10] = 1.0
    
    for _ in range(100):
        model.step()
    
    # u doit rester dans [0, 1]
    assert np.all(model.u >= 0.0), f"u < 0 détecté: min={model.u.min()}"
    assert np.all(model.u <= 1.0), f"u > 1 détecté: max={model.u.max()}"


# =============================================================================
# ATTAQUE 5: PRÉCISION NUMÉRIQUE
# =============================================================================
def test_float32_vs_float64_divergence():
    """Le modèle diverge-t-il entre float32 et float64?"""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.0}  # Pas de bruit pour comparer
    }
    
    model64 = Mem4ristorV2(config=cfg, seed=42)
    model64._initialize_params(N=100)
    
    # Convertir en float32
    model32 = Mem4ristorV2(config=cfg, seed=42)
    model32._initialize_params(N=100)
    model32.v = model32.v.astype(np.float32)
    model32.w = model32.w.astype(np.float32)
    model32.u = model32.u.astype(np.float32)
    
    for _ in range(1000):
        model64.step()
        model32.step()
    
    # La divergence doit rester raisonnable
    max_diff = np.max(np.abs(model64.v - model32.v.astype(np.float64)))
    assert max_diff < 0.1, f"Divergence numérique excessive: {max_diff:.6f}"


# =============================================================================
# ATTAQUE 6: HERETIC RATIO > 1.0 OU < 0
# =============================================================================
def test_heretic_ratio_above_one():
    """heretic_ratio > 1.0 est absurde"""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 1.5},  # > 1.0
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=100)
    
    # Tous les nœuds devraient être hérétiques
    heretic_count = model.heretic_mask.sum()
    assert heretic_count <= 100, f"Plus d'hérétiques que de nœuds: {heretic_count}"


def test_heretic_ratio_negative():
    """heretic_ratio < 0 devrait donner 0 hérétiques"""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': -0.1},  # NÉGATIF
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    model = Mem4ristorV2(config=cfg, seed=42)
    model._initialize_params(N=100)
    
    heretic_count = model.heretic_mask.sum()
    assert heretic_count == 0, f"Hérétiques avec ratio négatif: {heretic_count}"


# =============================================================================
# ATTAQUE 7: OVERFLOW/UNDERFLOW
# =============================================================================
def test_extreme_stimulus_overflow():
    """I_stimulus énorme"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=10)
    
    for _ in range(10):
        model.step(I_stimulus=1e10)  # ÉNORME
    
    assert np.all(np.isfinite(model.v)), f"Overflow détecté: v contient inf/nan"


def test_extreme_initial_conditions():
    """v initialisé à des valeurs extrêmes"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=10)
    
    model.v = np.full(10, 1e10)  # ÉNORME
    
    for _ in range(100):
        model.step()
    
    # Le système devrait revenir vers l'attracteur FHN
    assert np.all(np.abs(model.v) < 1e15), f"Le système n'a pas convergé: max(v)={np.max(np.abs(model.v))}"


# =============================================================================
# ATTAQUE 8: ENTRÉE DE MAUVAIS TYPE
# =============================================================================
def test_wrong_coupling_shape():
    """Matrice de couplage de mauvaise dimension"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=10)
    
    wrong_adj = np.ones((5, 5))  # Mauvaise taille
    
    with pytest.raises((ValueError, IndexError)):
        model.step(I_stimulus=0.0, coupling_input=wrong_adj)


def test_stimulus_wrong_size():
    """Stimulus vectoriel de mauvaise taille"""
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=10)
    
    wrong_stim = np.ones(5)  # Mauvaise taille
    
    with pytest.raises(ValueError):
        model.step(I_stimulus=wrong_stim)
