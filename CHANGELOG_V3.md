# CHANGELOG: v2.9.3 → v3.0.0

## Migration Summary

**Date**: 2026-02-16
**Auditor**: Claude Opus 4.6 (Automated Scientific Audit)
**Orchestrator**: Julien Chauvin
**Reason**: Scientific rigor audit identified overclaims, test weaknesses, and inconsistencies.

---

## Philosophy of Changes

This migration follows a single principle: **claims must match evidence**.
Every modification below either (a) corrects a factual inconsistency, (b) strengthens a weak test, or (c) recalibrates language to match what the code actually proves.

---

## Structural Changes

### 1. Core Engine: V3 Levitating Sigmoid becomes the canonical implementation
- **Before**: `core.py` contained `Mem4ristorV2` (linear kernel `1-2u`), `mem4ristor_v3.py` contained `Mem4ristorV3` (Levitating Sigmoid `tanh(π(0.5-u)) + δ` + Inhibition Plasticity)
- **After**: `core.py` contains `Mem4ristorV3` as the main class, incorporating all V2 security guards + V3 innovations
- **Rationale**: V3's smooth sigmoid eliminates the dead zone at u=0.5 where V2's linear kernel made coupling noise-dominated (LIMIT-01). Plasticity adds structural memory of dissidence.

### 2. Mem4ristorKing moved to experimental/
- **Before**: `src/mem4ristor/mem4ristor_king.py`
- **After**: `experimental/mem4ristor_king.py` with WARNING header
- **Rationale**: King has design issues (temporary state mutation, missing coupling_input passthrough) that make it unsuitable for production code. Kept as documented exploration.

### 3. Old mem4ristor_v3.py removed
- **Rationale**: Its code is now merged into `core.py`. Keeping it would create confusion.

---

## Claim Corrections (Preprint & Documentation)

| Original Claim | Correction | Justification |
|:---|:---|:---|
| "Law of 15%" | "Empirical Threshold of 15%" | Fails on scale-free networks (own LIMIT-02) |
| "topologically invariant" | "validated on regular lattices (2D grid, Small-World, Random)" | Scale-free counterexample exists |
| "CCC Validation" | "CCC Illustration" | No raw CCC data used; parameters hand-tuned to match known outcomes |
| Δt = 0.1 (preprint Limitations) | Δt = 0.05 (consistent everywhere) | Code default is 0.05; 0.1 was stale reference |
| Δt = 0.01 recommended (preprint) | Δt = 0.05 standard, 0.01 high-precision | Harmonized across code, config, and docs |

---

## Test Suite Hardening

| Test | Issue | Fix |
|:---|:---|:---|
| `test_nan_injection_survial` | Accepted `nan_count <= 1` | Changed to `nan_count == 0` |
| `test_manus_v2.py` (5 tests) | Used `print()` instead of `assert` | Converted to proper assertions |
| `test_snr_validity` | Didn't calculate SNR | Renamed and rewritten with actual SNR computation |
| `test_spatial_clustering` | Arbitrary threshold (<5) | Threshold justified by block size math |
| `test_two_unit_network_symmetry` | Tolerance of 1.0 | Reduced to 0.3 (reasonable for noise) |
| `test_float32_vs_float64` | Didn't test float32 ops | Removed (was testing nothing) |
| `test_fuzzing` | 50 iterations | Increased to 200, added input tracking |

---

## Benchmark Corrections

### benchmark_kuramoto.py
- **Before**: Mem4ristor on 2D grid vs Kuramoto all-to-all (topology mismatch)
- **After**: Both models use same adjacency matrix for fair comparison
- **Before**: Voltage→phase mapping via linear rescaling (physically invalid for FHN)
- **After**: Comparison uses entropy and variance directly (topology-agnostic metrics)

---

## Files Modified

- `src/mem4ristor/core.py` — Rewritten: V3 engine + V2 guards
- `src/mem4ristor/__init__.py` — Updated exports
- `src/mem4ristor/config.yaml` — Added V3 parameters
- `src/mem4ristor/mem4ristor_v3.py` — Removed (merged into core)
- `src/mem4ristor/mem4ristor_king.py` — Moved to experimental/
- `experimental/mem4ristor_king.py` — Added with WARNING
- `tests/test_kernel.py` — Hardened
- `tests/test_robustness.py` — Hardened
- `tests/test_fuzzing.py` — Hardened
- `tests/test_manus_v2.py` — Converted to proper assertions
- `experiments/benchmark_kuramoto.py` — Corrected methodology
- `docs/preprint.tex` — Claims recalibrated
- `README.md` — Updated for V3
- `pyproject.toml` — Version bump
- `VERSION` — 3.0.0
- `CITATION.cff` — Updated
- `CAFE-VIRTUEL-LIMITATIONS.md` — Updated status table
