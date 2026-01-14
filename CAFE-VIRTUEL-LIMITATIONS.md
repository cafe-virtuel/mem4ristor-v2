# LIMITATIONS (Scientific Truth Table)

| Claim | Status | Proven | Counter-Example | Falsifiable |
| :--- | :--- | :--- | :--- | :--- |
| u ∈ [0,1] works | ❌ UNPROVEN | No | u=0.6 → I_coup < noise | YES (see failure #12) |
| 15% threshold universal | ⚠️ EMPIRICAL | N=625 | Fails on Scale-Free Hubs | YES (see failure #13) |
| Hardware mapping | ❌ SPECULATION | None | No SPICE model exists | NO (unfalsifiable) |
| Long-term stability | ❌ FALSE | dt=0.05 | H drift > 5% @ 5000 steps | YES (see failures/) |
| Cross-platform parity | ✅ FIXED | Yes | MKL Non-determinism | YES (v2.6 fix) |

**Rule**: Any claim marked ❌ MUST be qualified as "Phenomenological" or "Speculative" in the preprint.

## Detailed Failure Inventory

### [FAIL-012] SNR Collapse at u > 0.5
When $u$ approaches 0.5, the signal $|(1-2u) \cdot D_{eff} \cdot L|$ approaches zero. If the noise floor $\sigma$ is fixed, there is a "Dead Zone" where Repulsive Social Coupling is purely noise-driven.
- **Proof**: Run `self_audit_v27.py` with noise=0.05.

### [FAIL-013] Topological Strangulation
In Scale-Free networks, conformist hubs can isolate heretics. If heretics are on the periphery, their phase-inversion signal is absorbed by high-degree conformists before reaching the whole network.
- **Proof**: `Attack 1` in `self_audit_v27.py`.

### [FAIL-014] RK45 Step-Size Artifacts
Even with RK45, extremely large networks ($N > 2500$) exhibit memory leaks or step-size collapse in `solve_ivp` if $I_{stimulus}$ is high-frequency.
- **Status**: Documented in `failures/stability_failure_N2500.log`.
