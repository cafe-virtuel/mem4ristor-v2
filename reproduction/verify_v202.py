import numpy as np
import os
import sys

# Ensure we can import the updated reference_impl
sys.path.append(os.path.dirname(__file__))
from reference_impl import Mem4ristorV2

def verify_v202():
    print("--- Mem4ristor v2.0.2 Verification ---")
    model = Mem4ristorV2()
    
    # 1. Baseline Phase (I=0)
    print("Running Baseline (800 steps)...")
    for _ in range(800):
        model.step(I_stimulus=0)
    
    h_baseline = model.calculate_entropy()
    std_baseline = np.std(model.v)
    print(f"Baseline (t=800): H={h_baseline:.4f}, std(v)={std_baseline:.4f}")
    
    # 2. Bias Phase (I=1.1)
    print("\nApplying Bias I=1.1 (2200 steps)...")
    entropies = []
    stds = []
    for _ in range(2200):
        model.step(I_stimulus=1.1)
        entropies.append(model.calculate_entropy())
        stds.append(np.std(model.v))
    
    mean_h_bias = np.mean(entropies[-500:]) # Look at terminal behavior
    mean_std_bias = np.mean(stds[-500:])
    
    print(f"Bias Phase Final Entropy: {mean_h_bias:.4f}")
    print(f"Bias Phase Final std(v): {mean_std_bias:.4f}")
    
    # VERDICT
    success = True
    if mean_h_bias <= 0.2:
        print("❌ FAIL: Entropy collapse detected (H <= 0.2)")
        success = False
    if mean_std_bias < 0.1:
        print("❌ FAIL: Insufficient divergence (std(v) < 0.1)")
        success = False
        
    if success:
        print("\n✅ SUCCESS: Mem4ristor v2.0.2 demonstrates stable diversity under bias.")
    else:
        print("\n❌ VERDICT: FAIL - v2.0.2 still exhibits synchronization issues.")

    print("\n--- Ablation Test: No Heretics ---")
    model_ablation = Mem4ristorV2()
    model_ablation.heretic_mask[:] = False # No heretics
    
    print("Running Baseline Ablation (800 steps)...")
    for _ in range(800):
        model_ablation.step(I_stimulus=0)
    
    print("Applying Bias I=1.1 to Ablated Model (2200 steps)...")
    for _ in range(2200):
        model_ablation.step(I_stimulus=1.1)
    
    h_final_ablation = model_ablation.calculate_entropy()
    std_final_ablation = np.std(model_ablation.v)
    print(f"Final state (t=3000) Ablated: H={h_final_ablation:.4f}, std(v)={std_final_ablation:.4f}")
    
    if h_final_ablation < 0.2:
        print("✅ SUCCESS: Ablation correctly shows collapse (H < 0.2)")
    else:
        print("❌ FAIL: Ablated model still has diversity. Mechanism not isolated!")

if __name__ == "__main__":
    verify_v202()
