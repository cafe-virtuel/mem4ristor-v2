import sys
import os
import numpy as np
import time

# Force insert source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mem4ristor.mem4ristor_king import Mem4ristorKing
from mem4ristor.symbiosis import CreativeProjector

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_chimera_demo():
    print_header("THE CHIMERA PROTOCOL (Mem4ristor V3 Final Demo)")
    print("Initializing the Organism...")
    
    # 1. Born: The Philosopher King (Phase 3 + V5 Core)
    # We use a config that enables Hysteresis (V5)
    config = {
        'hysteresis': {'enabled': True, 'theta_low': 0.35, 'theta_high': 0.65, 'fatigue_rate': 0.05}
    }
    chimera = Mem4ristorKing(config=config, seed=1337)
    
    # 2. Enlightened: The Creative Mind (Phase 4)
    mind = CreativeProjector(chimera, num_classes=5, seed=1337)
    
    print(f" Organism Born. Size: {chimera.N} neurons.")
    print(f" Initial Epsilon (Time Speed): {chimera.cfg['dynamics']['epsilon']}")
    print(f" Initial Sigma (Risk/Noise): {chimera.cfg['noise']['sigma_v']}")
    print("-" * 60)

    # --- SCENARIO 1: THE CREATIVE SPARK (Phase 4) ---
    print_header("SCENARIO 1: L'Apnée de l'Innovation (Creative Bias)")
    print("Concept: Transforming 'Trauma' (Resistance) into Inspiration.")
    
    # Inflict artificial trauma/knowledge
    chimera.w[:10] = 5.0 
    print(">> Inflicted 'Experience' (High Resistance) on first 10 neurons.")
    
    # Calculate Bias
    bias = mind.get_creative_bias(panic_level=0.5)
    print(f">> Generated Creative Bias Vector: {np.round(bias, 3)}")
    print("   (This vector pushes the Cortex away from the known trauma)")
    time.sleep(1)

    # --- SCENARIO 2: STABILITY & WATCHDOG (V5) ---
    print_header("SCENARIO 2: Le Gardien (Stability & Hysteresis)")
    print("Concept: Locking into a decision state to avoid flickering.")
    
    # Force undecided state
    chimera.u[:] = 0.5
    chimera.mode_state[:] = False # Sage mode
    
    print(">> Injecting weak stimulus...")
    chimera.step_with_governance(I_stimulus=0.1)
    print(f"   State: {'Fou (High Energy)' if np.all(chimera.mode_state) else 'Sage (Low Energy)'}")
    
    print(">> Injecting STRONG stimulus (Crossing Threshold)...")
    chimera.u[:] = 0.8 # Push above 0.65
    chimera._update_hysteresis()
    print(f"   State: {'Fou (High Energy)' if np.all(chimera.mode_state) else 'Sage (Low Energy)'}")
    print("   (System successfully LATCHED into High Energy state)")
    time.sleep(1)

    # --- SCENARIO 3: CONSTITUTIONAL CRISIS (Phase 3) ---
    print_header("SCENARIO 3: Le Roi Philosophe (Governance)")
    print("Concept: Frustration leads to Martial Law (Action over Democracy).")
    
    # Give an IMPOSSIBLE target to cause frustration
    print(">> Brainstorming on an impossible paradox...")
    impossible_target = np.full(chimera.N, 100.0) # Unreachable v
    
    for t in range(25):
        status = chimera.step_with_governance(target_vector=impossible_target)
        if t % 5 == 0:
            print(f"   Step {t}: Frustration Phi = {status['frustration']:.2f}")
            if status['martial_law']:
                print("   !!! MARTIAL LAW DECLARED !!! Doubt suspended. Forced Action.")
                break
                
    if not status['martial_law']:
        print("   (Warning: Martial Law threshold not reached in this short run)")
    else:
        print(">> System forced consensus to survive the paradox.")
    time.sleep(1)

    # --- SCENARIO 4: METACOGNITION (Phase 3) ---
    print_header("SCENARIO 4: L'Ennui & L'Autopoïèse")
    print("Concept: Changing own physics when bored.")
    
    # Force a boring state (Entropy = 0)
    print(">> Placing system in Sensory Deprivation Tank (Boredom)...")
    chimera.v[:] = 0.0
    
    initial_eps = chimera.cfg['dynamics']['epsilon']
    
    for t in range(50):
        chimera.step_with_governance()
        chimera.v[:] = 0.0 # Force stay boring
        
    final_eps = chimera.cfg['dynamics']['epsilon']
    print(f"   Initial Speed (Epsilon): {initial_eps}")
    print(f"   Final Speed (Epsilon):   {final_eps:.4f}")
    if final_eps > initial_eps:
        print(">> The system WOKE ITSELF UP (Increased metabolic rate due to boredom).")
    time.sleep(1)

    # --- SCENARIO 5: THE DREAMER (Phase 4) ---
    print_header("SCENARIO 5: La Nuit des Temps (Dreaming)")
    print("Concept: Hallucinating to consolidate memory.")
    
    print(">> Entering Night Mode (Stimulus = 0)...")
    dream_log = mind.dream_cycle(steps=10)
    
    print(f">> Generated {len(dream_log)} frames of Dream Content.")
    print(f"   Sample Dream Frame 0: {np.round(dream_log[0], 3)}")
    print("   (This content is purely generated from the 'Scars' of previous experiences)")
    
    print_header("DEMO COMPLETE: The Chimera is Alive.")

if __name__ == "__main__":
    run_chimera_demo()
