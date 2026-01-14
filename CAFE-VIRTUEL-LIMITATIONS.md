# Café Virtuel: Limitations & Known Failures

This document provides a radical disclosure of the technical and methodological boundaries of the Mem4ristor project. To preserve scientific integrity, we explicitly list cases where the model fails or where assumptions remain unvalidated.

## 1. Known Technical Failures

### 1.1 Entropy Drift (Euler Instability)
- **Status**: FAILED (at dt=0.05, N > 200)
- **Description**: In versions prior to v2.6, long-term simulations (>3000 steps) using the first-order Euler integrator exhibit a slow decay of diversity ($H$).
- **Remediation**: Transitioned to **RK45** (Scipy) in v2.6.

### 1.2 SNR Decay (Large Scale)
- **Status**: OBSERVED
- **Description**: As $N$ increases, the effective coupling $D_{eff} \propto 1/\sqrt{N}$ decreases. For $N > 1000$ and noise $\sigma \ge 0.05$, the signal of social repulsion can be masked by stochastic fluctuations, lead to noise-driven rather than physics-driven state transitions.

### 1.3 Topological Strangulation
- **Status**: PARTIALLY RESISTANT
- **Description**: In Scale-Free networks (Barabási-Albert) with high-degree hubs, conformist pressure on peripheral nodes is high. If heretics are concentrated in low-degree nodes, diversity is maintained but cannot easily trigger a global phase transition.

## 2. Theoretical Speculations

### 2.1 Hardware Mapping
- **Status**: UNVALIDATED (Conceptual)
- **Constraint**: There is currently NO physical SPICE model or HfO2 crossbar validation. The hardware mapping in the preprint is a **phenomenological proposal**.

### 2.2 Cubic Divisor ($D=4.0$)
- **Status**: EMPIRICAL
- **Constraint**: The choice of 4.0 or 5.0 for the cubic term is justified by sensitivity sweeps, not by a formal Lyapunov stability proof.

## 3. Methodological Constraints (Agentic Bias)

### 3.1 Defensive Confirmation
- **Description**: LLM-based agents (Antigravity) may exhibit a bias toward validating User hypotheses to be helpful. 
- **Safeguard**: Adversarial audit sessions with 'memory-less' red-team agents (Kimi) are required for version finalization.

### 3.2 Web Interface Persistence
- **Description**: Free-tier web interfaces used for Café Virtuel sessions lack persistent API-level constraint enforcement.
- **Safeguard**: Mandatory local `pre-commit.sh` audit and `failures/` logging.

---
*Documented on: 2026-01-14*
