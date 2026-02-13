# Mem4ristor V2: Neuromorphic Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18620597.svg)](https://doi.org/10.5281/zenodo.18620597)
[![Tests](https://github.com/Jusyl236/mem4ristor-v2/actions/workflows/test.yml/badge.svg)](https://github.com/Jusyl236/mem4ristor-v2/actions/workflows/test.yml)

**Mem4ristor V2** is a computational implementation of extended FitzHugh-Nagumo dynamics designed to investigate emergent critical states in neuromorphic networks. This research code focuses on the role of "Constitutional Doubt" ($u$) and "Structural Heretics" in preventing consensus collapse in scale-free and lattice networks.

> **Status**: v2.9.3 (Stable Research Release)

## 🔬 Key Scientific Features

*   **Constitutional Doubt ($u$):** A dynamic state variable that modulates coupling polarity based on local uncertainty, enabling repulsive social coupling when doubt is high.
*   **Structural Heretics:** A subset of nodes with inverted stimulus perception, critical for maintaining global diversity (Empirically validated at ~15%).
*   **Scale-Invariant Dynamics:** Normalized coupling strength ($D_{eff} = D/\sqrt{N}$) ensures consistent behavior across network sizes ($N=10$ to $N=2500$).

## 🚀 Installation

This project is structured as a standard Python package.

```bash
git clone https://github.com/Jusyl236/mem4ristor-v2.git
cd mem4ristor-v2
pip install -e .
```

*Note: The `-e` flag installs in editable mode, allowing you to modify source code without reinstalling.*

## 💻 Usage

### Quick Start (Python API)

```python
from mem4ristor.core import Mem4Network

# Initialize a network (N=100, 15% Heretics)
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)

# Run simulation for 1000 steps
for step in range(1000):
    net.step(I_stimulus=0.5)

# Calculate final entropy (measure of diversity)
print(f"Final System Entropy: {net.calculate_entropy():.4f}")
```

### Running Benchmarks

To verify the stochastic variability and stability of the system, run the included Monte Carlo benchmark:

```bash
python experiments/benchmark_variability.py
```

*Expected output: Entropy $H \approx 1.45 \pm 0.10$ indicating a critical regime.*

## ⚙️ Configuration

The model is highly configurable via `src/mem4ristor/config.yaml`. You can adjust:

*   **Dynamics:** `a`, `b`, `epsilon` (FHN parameters)
*   **Coupling:** `D` (Strength), `heretic_ratio`
*   **Doubt:** `epsilon_u`, `u_clamp`
*   **Noise:** `sigma_v`

## 🧪 Testing

The repository includes a comprehensive test suite using `pytest`.

```bash
# Run all tests
pytest

# Run only robustness tests
pytest tests/test_robustness.py
```

## 📂 Repository Structure

*   `src/mem4ristor/`: Core package source code.
*   `tests/`: Unit and robustness tests.
*   `experiments/`: Benchmark scripts and SPICE netlists.
*   `docs/`: Scientific documentation and preprint.

## 📜 Citation

If you use this code in your research, please cite the associated dataset/preprint:

```bibtex
@software{mem4ristor_v2,
  author       = {Julien Chauvin},
  title        = {Mem4ristor V2: Neuromorphic Cognitive Architecture},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18620597},
  url          = {https://doi.org/10.5281/zenodo.18620597}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
