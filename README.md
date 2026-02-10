# Mem4ristor v2.9.3 (Antigravity Hardened Core)
Implementation Kit (Computational Model)

> **Status**: Attractor Diversity Research Prototype (**v2.9.3**)  
> **Origin**: [Café Virtuel](https://www.cafevirtuel.org) Collaboration  
> **Author**: Julien Chauvin ([ORCID: 0009-0006-9285-2308](https://orcid.org/0009-0006-9285-2308))  
> **Concept**: Neuromorphic cognitive resistance through **constitutional doubt** and **structural heretics**.

## 🔬 Overview
Mem4ristor v2.9.3 is a **neuromorphic-inspired computational model** designed to investigate algorithmic uniformization (consensus collapse). Unlike classical oscillators, it integrates a dynamic doubt variable ($u$) and structural heretics to maintain deliberative diversity in simulated environments.

This repository provides the "computational specification" for the phenomenological model described in the [accompanying preprint](docs/preprint.tex).

### 🛡️ Scientific Rationale (Addressing Consensus Collapse)
The Mem4ristor architecture introduces three mechanisms to prevent "Deep Time" synchronization:
1. **The Doubt Kernel $(1-2u)$**: Dynamically modulates coupling polarity. When local uncertainty $u$ crosses the 0.5 threshold, the unit switches from attractive to **repulsive social coupling**.
2. **Structural Heretics (15%)**: A fixed sub-population wired with inverted stimulus perception. This 15% ratio serves as a critical threshold for maintaining global diversity under bias.
3. **Formal Robustness**: The "Constitutional Doubt" mechanism acts as a cognitive buffer, allowing the network to resist forced consensus through phase inversion.

## 📂 Repository Structure
- `src/mem4ristor/`: Cœur algorithmique (Moteur v2.6 - Simulation).
- `docs/`: Preprint LaTeX, figures et manuscrit PDF.
- `experiments/`: 
    - `protocol/`: Script principal de démonstration scientifique (`run_protocol_v26.py`).
    - `benchmark_kuramoto.py`: Comparative benchmark vs Standard/Frustrated Kuramoto.
    - `phase_diagram.py`: Phase diagram H(heretic_ratio, D) heatmap.
    - `lyapunov_numerical.py`: Numerical Lyapunov candidate verification.
    - `robustness/`: Tests de résilience (Topologie, Scale-Free, Sweep, Résurrection).
    - `spice/`: ngspice behavioral simulation netlist.
- `results/`: Sorties automatisées (Benchmarks, Phase Diagram, Figures).
- `requirements.txt`: Python dependencies.

## 🚀 Quick Start (Verify in 5 minutes)

### 1. Requirements
- Python 3.8+
- `pip install -r requirements.txt`

### 2. Run the Formal Verification Protocol (Recommended)
```bash
python experiments/protocol/run_protocol_v26.py
```
*Note: This script provides an interactive demonstration of all robustness tests (Percolation, Topology, Resurrection).*

### 3. Expected Results (Standard Signatures v2.6)
- **Diversity Percolation**: Significant entropy jump precisely at $\eta=0.15$.
- **Universal Robustness**: Stable diversity on Small-World and Random networks.
- **Active Resurrection**: Immediate symmetry breaking from consensus ($H=0 \to H>1.5$).

## 🔌 Architectural Mapping (Conceptual)
The model provides a high-level **conceptual mapping** for potential memristive implementations:
- **Recovery variable ($w$)**: Analogous to memristor conductance.
- **Doubt ($u$)**: Modeled as local accumulation/dissipation processes.
- **Physics**: Interpreted as **Frustrated Synchronization**.

---

## ❓ FAQ (Sincerity Disclosure)

**Is this a real neuromorphic hardware design?**  
No. This is a **Python simulation** of a phenomenological model. The hardware mappings and SPICE mentions are conceptual explorations of how this logic *could* be implemented in HfO2 crossbars.

**Who wrote this?**  
This project is developed through the **Café Virtuel** methodology, involving a human researcher (The Barman) and a collaborative ensemble of Large Language Models (LLMs). Every line of code and every claim is traceable in the [Café Virtuel repository](https://github.com/Jusyl236/Cafe-Virtuel).

**Is the "Law of 15%" a physical law?**  
It is an **empirical observation** within our computational framework. We observed a transition from consensus to diversity specifically at this ratio. It is a property of the model's dynamics under the tested configurations.

---
**Café Virtuel** - *From intuition to formal specification.*
