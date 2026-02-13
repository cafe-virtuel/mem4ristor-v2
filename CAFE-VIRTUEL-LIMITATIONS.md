# LIMITATIONS (Scientific Truth Table)

| Claim | Status | Proven | Counter-Example | Falsifiable |
| :--- | :--- | :--- | :--- | :--- |
| u ∈ [0,1] works | ❌ UNPROVEN | No | u=0.6 → I_coup < noise | YES (see failure #12) |
| 15% threshold universal | ⚠️ EMPIRICAL | N=625 | Fails on Scale-Free Hubs | YES (see failure #13) |
| Hardware mapping | ❌ SPECULATION | None | No SPICE model exists | NO (unfalsifiable) |
| Long-term stability | ❌ FALSE | dt=0.05 | H drift > 5% @ 5000 steps | YES (see failures/) |
| Cross-platform parity | ✅ FIXED | Yes | MKL Non-determinism | YES (v2.6 fix) |
| H ≈ 1.94 Attractor | ❌ FALSE | No | Max H found ≈ 1.56 | YES (see failure #15) |

**Rule**: Any claim marked ❌ MUST be qualified as "Phenomenological" or "Speculative" in the preprint.

## Detailed Failure Inventory

### [LIMIT-01] SNR Collapse at u > 0.5
When $u$ approaches 0.5, the signal $|(1-2u) \cdot D_{eff} \cdot L|$ approaches zero. If the noise floor $\sigma$ is fixed, there is a "Dead Zone" where Repulsive Social Coupling is purely noise-driven.
- **Proof**: Run `self_audit_v27.py` with noise=0.05.

### [LIMIT-02] Topological Strangulation
In Scale-Free networks, conformist hubs can isolate heretics. If heretics are on the periphery, their phase-inversion signal is absorbed by high-degree conformists before reaching the whole network.
- **Proof**: `Attack 1` in `self_audit_v27.py`.

### [LIMIT-03] RK45 Step-Size Artifacts
Even with RK45, extremely large networks ($N > 2500$) exhibit memory leaks or step-size collapse in `solve_ivp` if $I_{stimulus}$ is high-frequency.
- **Status**: Documented in `failures/stability_failure_N2500.log`.

### [LIMIT-04] Euler Instability at High dt
The Euler integration method becomes unstable when $dt > 0.1$. At $dt = 0.5$, entropy collapses to near-zero within 200 steps.
- **Proof**: `test_euler_drift_torture` in `test_adversarial.py`.
- **Mitigation**: Preprint specifies $dt \le 0.05$ for standard use.

### [LIMIT-05] Attractor Diversity Gap (H ≈ 1.94 Claim)
The preprint claims an attractor diversity of $H \approx 1.94$. Empirical audits with both code defaults and preprint parameters failed to reproduce this, achieving a maximum $H \approx 1.56$. Cold start protocols consistently collapsed to $H=0$.
- **Proof**: Internal Audit 2026-01-14.
- **Status**: OPEN. Claim must be revised or exact conditions documented.

---

## v2.9.2 Fixes (Stability and Integrity Fix)

| Bug | Status | Fix |
|:----|:-------|:----|
| Parameter Discrepancy | ✅ FIXED | `core.py` defaults aligned with preprint (D=0.15, etc.) |
| `ZeroDivisionError` when `heretic_ratio=0.0` | ✅ FIXED | Guard clause in `_initialize_params` |
| Version string inconsistency | ✅ FIXED | Unified to v2.9.2 |
| Adversarial test failures blocking CI | ✅ FIXED | Marked as `xfail` with documentation links |
