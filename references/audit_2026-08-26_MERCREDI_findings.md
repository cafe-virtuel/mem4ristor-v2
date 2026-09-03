# MERCREDI 2026-08-26 — Tests pytest findings

**Commit audite:** 9995db6 (V6.0.0, working tree EN SYNC avec origin/main, 0 commit de retard)
**Branche:** main
**Interprete Python:** C:/Users/julch/AppData/Local/Programs/Python/Python313/python.exe (numpy 2.2.6 OK)

## Resultats

### pytest tests/ -q --tb=no
- **151 passed, 2 xfailed, 3 warnings in 15.79s**
- Exit code: 0 (succes)

### xfail (attendu, NON regression)
- `tests/test_adversarial.py::test_snr_significance_breakdown` — Known theoretical limitation: SNR collapse at high noise regimes.
- `tests/test_adversarial.py::test_euler_drift_torture` — Known numerical instability at dt > 0.1. Recommended dt <= 0.05.

### Warnings (benins)
- `tests/test_fuzzing.py::test_vicious_fuzzing_inputs` : RuntimeWarning overflow (dyn `v**3` et `w_ratio**2`) — attendu en fuzzing extreme
- `tests/test_kernel.py::test_rk45_stability` : RuntimeWarning RK45 + sigma_v (incoherence adaptive/stochastic documentee)

### Inventaire test files (pytest --collect-only)
21 fichiers, 153 tests collectes :
- test_adversarial.py: 3
- test_complex_doubt.py: 7
- test_consolidation_watchdog.py: 3
- test_coordination_metrics.py: 14
- test_directed_guard.py: 2
- test_fuzzing.py: 2
- test_kernel.py: 8
- test_manus_v2.py: 5
- test_robustness.py: 15
- test_scientific_regression.py: 10
- test_sigma_social_override.py: 6
- test_symbiosis_creativity.py: 2
- test_symbiosis_swarm.py: 1
- test_u_clamp_invariant.py: 9
- test_v4_extensions.py: 17
- test_v5_art.py: 10
- test_v5_compartments.py: 9
- test_v5_hysteresis.py: 3
- test_v5_metacognitive.py: 6
- test_v5_nonlocal_coupling.py: 9
- test_version_consistency.py: 12

### Cross-check README vs tests
- README ligne 110 : "comprehensive test suite using pytest" (pas de compte specifique) — pas d'incoherence sur le compte exact
- README ligne 117 : `pytest tests/test_robustness.py` (existe ✅, 15 tests)
- README ligne 120 : `pytest tests/test_scientific_regression.py` (existe ✅, 10 tests)
- Aucun chemin de test cite ne manque

### Comparaison baselines historiques
- 2026-06-08 : baseline spec "92 tests"
- 2026-06-11 : "84 passed, 2 xfailed" (mesure reelle)
- 2026-06-20 : "84 passed, 2 xfailed" (mesure reelle confirmee)
- 2026-08-25 (MARDI) : "27/27 exports resolues" (etat post-reorg V6.0.0)
- **2026-08-26 (MERCREDI) : 151 passed, 2 xfailed** ← progression majeure depuis le reorg V6.0.0 (84 → 151, +80%)

### Conclusion
**OK — 151 passed / 2 xfailed, aucune regression, 21 fichiers de test, V6.0.0 maintenu.**

— Hermes Agent / Cafe Virtuel
