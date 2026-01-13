import numpy as np
import os
import sys
import yaml

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def run_standard_verification():
    print("🛡️ MEMRISTOR v2.0.3 | INDUSTRIAL VERIFICATION PROTOCOL")
    print("-------------------------------------------------------")
    
    success = True
    
    # --- PHASE 1: PASSIVE RESILIENCE (Random IC) ---
    print("\n[PHASE 1] PASSIVE RESILIENCE (Random ICs)")
    model_rand = Mem4ristorV2()
    # Baseline
    for _ in range(800): model_rand.step(0.4)
    # Bias
    for _ in range(1000): model_rand.step(1.1)
    
    h_rand = model_rand.calculate_entropy()
    print(f"Terminal Entropy (Random IC): {h_rand:.4f}")
    if h_rand > 1.2:
        print("✅ PASS: Diversity maintained in standard regime.")
    else:
        print("❌ FAIL: Diversity collapse in standard regime.")
        success = False
        
    # --- PHASE 2: ACTIVE RESTORATION (Cold Start / Homogeneous IC) ---
    print("\n[PHASE 2] ACTIVE RESTORATION (Cold Start Protocol)")
    model_cold = Mem4ristorV2()
    # Force full consensus (Cold Start)
    model_cold.v[:] = 0.0
    model_cold.w[:] = 0.0
    model_cold.u[:] = 0.05
    
    print("Initializing from zero entropy (H=0.0000)...")
    print(f"Heretics count: {model_cold.heretic_mask.sum()}")
    h_idx = np.where(model_cold.heretic_mask)[0][0]
    n_idx = np.where(~model_cold.heretic_mask)[0][0]
    
    h_cold_list = []
    std_cold_list = []
    
    for i in range(2500): 
        # Calculate coupling for heretic before step to trace it
        delta_v = model_cold.adj @ model_cold.v - model_cold.v
        i_coup_h = model_cold.D_eff * (1.0 - model_cold.u[h_idx]) * delta_v[h_idx]
        
        model_cold.step(1.1)
        
        if i >= 2000:
            ent = model_cold.calculate_entropy()
            h_cold_list.append(ent)
            std_cold_list.append(np.std(model_cold.v))
            if i >= 2450:
                print(f"Step {i} | H: {ent:.4f} | u[h]: {model_cold.u[h_idx]:.4f} | I_coup[h]: {i_coup_h:.4f}")
        if i % 250 == 0:
            print(f"Step {i:4d} | v[n]: {model_cold.v[n_idx]:.4f} | v[h]: {model_cold.v[h_idx]:.4f} | u[h]: {model_cold.u[h_idx]:.4f}")



    h_cold = np.mean(h_cold_list)
    std_cold = np.mean(std_cold_list)
    print(f"Final Average Entropy (Cold Start): {h_cold:.4f}")
    print(f"Final Average std(v): {std_cold:.4f}")
    
    # Threshold 0.1: Non-zero divergence proving symmetry-breaking
    if h_cold > 0.1 and std_cold > 0.05:
        print("✅ PASS: Mechanism successfully broke consensus symmetry (Resurrection).")
    else:
        print("❌ FAIL: Mechanism failed to break symmetry from cold start.")
        success = False

        
    # --- PHASE 3: CAUSAL ISOLATION (Ablation under Cold Start) ---
    print("\n[PHASE 3] CAUSAL ISOLATION (Ablation Study)")
    model_abl = Mem4ristorV2()
    model_abl.v[:] = 0.0
    model_abl.w[:] = 0.0
    model_abl.heretic_mask[:] = False # Remove the CLAIMED mechanism
    
    for _ in range(2500): model_abl.step(1.1)

    
    h_abl = model_abl.calculate_entropy()
    print(f"Terminal Entropy (Ablated): {h_abl:.4f}")
    
    if h_abl < 0.1:
        print("✅ PASS: Mechanism isolation verified. Collapse occurs without heretics.")
    else:
        print("❌ FAIL: Spurious diversity detected in ablated model!")
        success = False
        
    print("\n-------------------------------------------------------")
    if success:
        print("🏆 VERDICT: MEM4RISTOR v2.0.3 CERTIFIED (Industrial Ready)")
    else:
        print("🚫 VERDICT: AUDIT FAILED - Fix required.")
        sys.exit(1)

if __name__ == "__main__":
    run_standard_verification()
