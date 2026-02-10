"""
Benchmark Comparatif : Mem4ristor vs Kuramoto (Standard & Frustré)
=================================================================
Compare la préservation de diversité entre 3 modèles en utilisant
des métriques NORMALISÉES pour une comparaison équitable :

  1. Order Parameter R ∈ [0,1] : R=1 = sync totale, R≈0 = désordre complet
     - Kuramoto: R = |1/N Σ exp(i·θ_j)|
     - Mem4ristor: R = |1/N Σ exp(i·2π·(v_j - v_min)/(v_max - v_min))|
     
  2. Shannon Entropy H (chaque modèle avec ses propres bins naturels)
  
  3. Diversity Index D = (nombre d'états uniques) / (nombre max d'états)

Produit :
  - CSV de résultats dans results/
  - Graphe R(t) et H(t) superposés (PNG)

Usage :
  python experiments/benchmark_kuramoto.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network


def compute_order_parameter_phase(theta):
    """Kuramoto order parameter: R = |<exp(iθ)>|. R=1 = synchronized."""
    return np.abs(np.mean(np.exp(1j * theta)))


def compute_order_parameter_v(v):
    """
    Generalized order parameter for voltage-based models.
    Map v to phase on unit circle, then compute R.
    """
    v_range = np.max(v) - np.min(v)
    if v_range < 1e-10:
        return 1.0  # All identical = fully synchronized
    phases = 2 * np.pi * (v - np.min(v)) / v_range
    return np.abs(np.mean(np.exp(1j * phases)))


def shannon_entropy(values, n_bins=5):
    """Shannon entropy with uniform binning."""
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-10:
        return 0.0
    counts, _ = np.histogram(values, bins=n_bins, range=(vmin - 0.01, vmax + 0.01))
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ─────────────────────────────────────────────────────────
# 1. Kuramoto Standard
# ─────────────────────────────────────────────────────────
def simulate_kuramoto_standard(N, K, steps, dt=0.01, seed=42):
    """dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i). Should synchronize (R→1)."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, N)
    omega = rng.normal(0, 1.0, N)
    
    R_history, H_history = [], []
    
    for _ in range(steps):
        sin_diff = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])
        coupling = (K / N) * np.sum(sin_diff, axis=1)
        theta += (omega + coupling) * dt
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        
        R_history.append(compute_order_parameter_phase(theta))
        H_history.append(shannon_entropy(theta, n_bins=5))
    
    return R_history, H_history


# ─────────────────────────────────────────────────────────
# 2. Kuramoto Frustré
# ─────────────────────────────────────────────────────────
def simulate_kuramoto_frustrated(N, K, alpha, frustration_ratio, steps, dt=0.01, seed=42):
    """dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i + α_ij). Partial sync."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, N)
    omega = rng.normal(0, 1.0, N)
    
    frustration_matrix = np.zeros((N, N))
    n_frustrated = int(frustration_ratio * N * N)
    frustrated_indices = rng.choice(N * N, n_frustrated, replace=False)
    frustration_matrix.flat[frustrated_indices] = alpha
    
    R_history, H_history = [], []
    
    for _ in range(steps):
        phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis] + frustration_matrix
        coupling = (K / N) * np.sum(np.sin(phase_diff), axis=1)
        theta += (omega + coupling) * dt
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        
        R_history.append(compute_order_parameter_phase(theta))
        H_history.append(shannon_entropy(theta, n_bins=5))
    
    return R_history, H_history


# ─────────────────────────────────────────────────────────
# 3. Mem4ristor
# ─────────────────────────────────────────────────────────
def simulate_mem4ristor(size, heretic_ratio, steps, seed=42):
    """Mem4ristor v2.9.3. Should maintain diversity (R < 1, H > 0)."""
    net = Mem4Network(size=size, heretic_ratio=heretic_ratio, seed=seed,
                      boundary='periodic')
    
    R_history, H_history = [], []
    
    for _ in range(steps):
        net.step(I_stimulus=0.0)
        R_history.append(compute_order_parameter_v(net.v))
        H_history.append(shannon_entropy(net.v, n_bins=5))
    
    return R_history, H_history


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  BENCHMARK : Mem4ristor vs Kuramoto (Normalized Comparison)")
    print("=" * 65)
    
    N = 100
    SIZE = 10
    STEPS = 1000
    K = 2.0
    ALPHA = np.pi / 4
    FRUST_RATIO = 0.15
    HERETIC_RATIO = 0.15
    
    print(f"\n  N={N}, Steps={STEPS}, K={K}")
    print(f"  Kuramoto frustré: α={ALPHA:.2f}, ratio={FRUST_RATIO}")
    print(f"  Mem4ristor: heretic_ratio={HERETIC_RATIO}\n")
    
    # Run
    print("  [1/3] Kuramoto Standard...", end=" ", flush=True)
    R_std, H_std = simulate_kuramoto_standard(N, K, STEPS)
    print(f"R={R_std[-1]:.3f}, H={H_std[-1]:.3f}")
    
    print("  [2/3] Kuramoto Frustré...", end=" ", flush=True)
    R_frust, H_frust = simulate_kuramoto_frustrated(N, K, ALPHA, FRUST_RATIO, STEPS)
    print(f"R={R_frust[-1]:.3f}, H={H_frust[-1]:.3f}")
    
    print("  [3/3] Mem4ristor v2.9.3...", end=" ", flush=True)
    R_mem, H_mem = simulate_mem4ristor(SIZE, HERETIC_RATIO, STEPS)
    print(f"R={R_mem[-1]:.3f}, H={H_mem[-1]:.3f}")
    
    # ── Results ──
    print("\n" + "─" * 65)
    print(f"  {'Model':<25} {'R_final':>8} {'R_mean':>8} {'H_final':>8} {'H_mean':>8}")
    print("─" * 65)
    
    models = [
        ("Kuramoto Standard", R_std, H_std),
        ("Kuramoto Frustré", R_frust, H_frust),
        ("Mem4ristor v2.9.3", R_mem, H_mem),
    ]
    
    for name, R, H in models:
        print(f"  {name:<25} {R[-1]:>8.3f} {np.mean(R[100:]):>8.3f}"
              f" {H[-1]:>8.3f} {np.mean(H[100:]):>8.3f}")
    
    # Key metric: Lower R = more diversity = better at resisting synchronization
    print("\n  📌 Interprétation :")
    print("     R proche de 1 = synchronisation totale (ÉCHEC de la diversité)")
    print("     R proche de 0 = diversité maximale (SUCCÈS)\n")
    
    if R_mem[-1] < R_std[-1]:
        print("  ✅ Mem4ristor résiste MIEUX à la synchronisation que Kuramoto Standard")
    else:
        print("  ⚠️ Mem4ristor se synchronise autant que Kuramoto Standard")
    
    if R_mem[-1] < R_frust[-1]:
        print("  ✅ Mem4ristor résiste MIEUX que le Kuramoto Frustré")
    else:
        print("  ⚠️ Le Kuramoto Frustré résiste autant ou mieux")
    
    # ── CSV ──
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'benchmark_kuramoto.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'R_std', 'R_frust', 'R_mem4', 'H_std', 'H_frust', 'H_mem4'])
        for t in range(STEPS):
            writer.writerow([t, R_std[t], R_frust[t], R_mem[t], 
                           H_std[t], H_frust[t], H_mem[t]])
    print(f"\n  📊 CSV: {os.path.abspath(csv_path)}")
    
    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t = np.arange(STEPS)
    
    # R(t) — Order Parameter
    ax1.plot(t, R_std, color='#e74c3c', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Std (R={R_std[-1]:.2f})')
    ax1.plot(t, R_frust, color='#f39c12', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Frustré (R={R_frust[-1]:.2f})')
    ax1.plot(t, R_mem, color='#2ecc71', alpha=0.9, linewidth=2.0,
             label=f'Mem4ristor (R={R_mem[-1]:.2f})')
    
    ax1.set_ylabel('Order Parameter R', fontsize=12)
    ax1.set_title('Synchronization Resistance: Mem4ristor vs Kuramoto', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
    ax1.axhline(y=0.0, color='green', linestyle=':', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.annotate('← Full Sync (bad)', xy=(50, 0.95), fontsize=9, color='red', alpha=0.6)
    ax1.annotate('← Full Diversity (good)', xy=(50, 0.05), fontsize=9, color='green', alpha=0.6)
    
    # H(t) — Shannon Entropy (uniform bins for fair comparison)
    ax2.plot(t, H_std, color='#e74c3c', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Std (H={H_std[-1]:.2f})')
    ax2.plot(t, H_frust, color='#f39c12', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Frustré (H={H_frust[-1]:.2f})')
    ax2.plot(t, H_mem, color='#2ecc71', alpha=0.9, linewidth=2.0,
             label=f'Mem4ristor (H={H_mem[-1]:.2f})')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Shannon Entropy H (bits)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    
    fig_path = os.path.join(results_dir, 'benchmark_kuramoto.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  📈 Figure: {os.path.abspath(fig_path)}")
    
    print("\n" + "=" * 65)
    print(f"  Benchmark terminé — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == '__main__':
    main()
