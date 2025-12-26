# Mem4ristor v2.0.4.1 (Verification Suite)
Implementation Kit

> **Status**: Attractor Diversity Research Prototype (v2.0.4.1)  


> **Origin**: Café Virtuel Collaboration  
> **Concept**: Neuromorphic ethical resistance through constitutional doubt and structural heretics.

## 🔬 Overview
Mem4ristor v2.0 is a **neuromorphic cognitive primitive** designed to investigate algorithmic uniformization (consensus collapse). Unlike classical oscillators, it integrates a dynamic doubt variable ($u$) and structural heretics to maintain deliberative diversity in simulated environments.

This repository provides the "executable specification" for the model described in the accompanying preprint.

## 📂 Repository Structure
- `docs/preprint.tex`: The scientific specification and hardware mapping.
- `src/mem4ristor/`: Core packaged logic for the Mem4ristor v2.0 engine (Production/Optimized).
- `reproduction/`: 
    - **`reference_impl.py`**: The **Core Functional Specification** (Ground truth).
    - `CONFIG_DEFAULT.yaml`: The "Golden Run" parameter set (No-touch reference).
    - `tests_reproduce_paper.py`: Automated benchmark and ablation study suite.
    - `results/`: Directory for reproduction outputs/plots.
- `examples/grok_hardware_sim.py`: Comparative simulation vs. classical models.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Standard GitHub cleanup.

## 🚀 Quick Start (Verify in 5 minutes)

### 1. Requirements
- Python 3.8+
- `pip install -r requirements.txt`

### 2. Run the Golden Run
```bash
python reproduction/tests_reproduce_paper.py
```

### 3. Expected Results (Golden Signatures v2.0.3)
A lab-standard reproduction using $N=100$ and default parameters is expected to encounter these empirical targets:
- **Baseline Diversity (Random IC)**: $H > 1.80$.
- **Steady-State Attractor Stability (Long Term)**: $H_{avg} > 1.8$ (verified for $>50,000$ iterations).
- **Causally Dominant Mechanism**: Diversification is maintained under IC-Compression ($W \to 0$), whereas ablated control models undergo irreversible collapse.
- **Multimodal Diversity Score (MDS)**: $H \times (N_{\text{occupied}} / 5) > 0.40$ in tested bias regimes ($I_{stim} > 1.0$).
- **Active Restoration (Homogeneous IC)**: Starting from $v=0, w=0$, heretic units restore diversity to **$H > 0.60$** within 1500 steps under bias.
- **Bias Resilience**: Under $I_{stim} = 1.1$, the system maintains **$H(t) \approx 1.90$**.
- **Causal Isolation**: In the Homogeneous IC regime, ablation of heretics yields $H \equiv 0$.
- **Epistemic Stability**: Mean doubt $\bar{u} = 0.05 \pm 0.01$.


*Note: Exact values depend on noise realization and initial conditions; bounds are empirical.*

> [!IMPORTANT]
> **Evolutionary Note**: v2.0.3 solved the "Cold Start" problem. v2.0.4 solves the "Deep Time" problem, where the system previously suffered from periodic rhythmic collapses. By introducing active repulsion and phase de-synchronization, we have achieved a **Permanently Stable Cognitive assembly**.


## 🔌 Hardware Mapping (Architectural BOM)
The model provides a high-level **mapping for HfO$_2$-based memristive crossbars**. This is an architectural specification for neuromorphic designers, not a ready-to-print PCB kit:
- **Recovery variable ($w$)**: Encoded in memristor conductance.
- **Doubt ($u$)**: Implemented as a local accumulation/dissipation circuit.
- **Heretics**: Wired with inverted coupling polarity.
- **Tolerance**: Circuit stability requires $\Delta V_{noise} < 10\%$.

## 📜 License & Citation
Mem4ristor v2.0 is released as an **Open Research Project** by the Café Virtuel.  
*Citation*: "Mem4ristor v2.0: A Doubt-Based Neuromorphic Architecture Resisting Algorithmic Uniformity", Julien Chauvin et al., 2025.

---
**Café Virtuel** - *Transforming intuitions into executable reality.*
