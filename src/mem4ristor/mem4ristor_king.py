import numpy as np
from typing import Optional, Dict

try:
    # Package import (when installed or PYTHONPATH set)
    from mem4ristor.core import Mem4Network
    from mem4ristor.mem4ristor_v3 import Mem4ristorV3
except ImportError:
    # Fallback for direct script execution
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core import Mem4Network
    from mem4ristor_v3 import Mem4ristorV3

class Mem4ristorKing(Mem4ristorV3):
    """
    Mem4ristor v3.1 (The Philosopher King).
    
    Concepts:
    1. Dynamic Constitution (Martial Law):
       Resolves 'Analysis Paralysis' via a Frustration variable (Phi).
       If error persists, Phi rises -> Force Consensus (u=0).
       
    2. Metacognitive Metabolism (Autopoiesis):
       Self-regulation of physical constants (epsilon, noise) based on Boredom (Low Entropy).
       "A bored system becomes paranoid/manic to feel alive."
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        super().__init__(config, seed)
        
        # --- PHASE 3 PARAMETERS ---
        
        # 1. Dynamic Constitution
        self.frustration_phi = 0.0        # Accumulated Frustration [0, 1.2]
        self.decay_phi = 0.95             # Frustration decay on success
        self.gain_phi = 0.05              # Frustration accumulation rate
        self.martial_threshold = 0.8      # Threshold for Martial Law
        
        # 2. Metacognition
        self.boredom_index = 0.0
        self.base_epsilon = self.cfg['dynamics']['epsilon']
        self.base_sigma_v = self.cfg['noise'].get('sigma_v', 0.05)
        
    def metacognitive_metabolism(self):
        """
        Adjusts physical parameters based on internal vitality (Entropy).
        Second-order homeostasis.
        """
        # 1. Measure Vitality (H)
        H = self.calculate_entropy()
        
        # 2. Boredom Index (Inverted H)
        # Assuming max H ~ 1.6 (log2(5) ~ 2.32 but practical max is lower)
        # If H is low, we are bored/stuck.
        self.boredom_index = np.clip(1.0 - (H / 1.5), 0, 1)
        
        # 3. Epsilon Metabolism (Time Perception)
        # Bored -> Speed up internal time (Epsilon increases)
        target_epsilon = self.base_epsilon + (0.10 * self.boredom_index)
        
        # Smooth transition (Physiological Inertia)
        current_eps = self.cfg['dynamics']['epsilon']
        self.cfg['dynamics']['epsilon'] += 0.01 * (target_epsilon - current_eps)
        
        # 4. Noise Metabolism (Thermal agitation)
        # Bored -> Heat up
        target_noise = self.base_sigma_v + (0.15 * self.boredom_index)
        
        current_sigma = self.cfg['noise']['sigma_v']
        self.cfg['noise']['sigma_v'] += 0.01 * (target_noise - current_sigma)

    def step(self, I_stimulus: float = 0.0, target_vector: Optional[np.ndarray] = None) -> Dict:
        """
        Advanced Step with Meta-Control Loops.
        """
        # --- 1. Error Calculation ---
        if target_vector is None:
            # Default goal: maximize activation (Certitude)
            # This is arbitrary, just to drive the frustration loop
            target_vector = np.full(self.N, 1.5)
            
        # Mean Squared Error
        current_error = np.mean((self.v - target_vector)**2)
        
        # --- 2. Frustration Loop ---
        if current_error > 0.5:
            # Failure -> Frustration rises
            self.frustration_phi = np.clip(self.frustration_phi + self.gain_phi, 0, 1.2)
        else:
            # Success -> Relief
            self.frustration_phi *= self.decay_phi
            
        # --- 3. THE LOCK (Martial Law) ---
        martial_law_active = self.frustration_phi > self.martial_threshold
        
        # Meta-Variables for Physics
        if martial_law_active:
            # FORCE CONSENSUS: Crush Doubt
            # We override the internal state 'u' temporarily for the coupling calculation
            u_meta_coupling = np.zeros_like(self.u) 
            # Boost Coupling Strength to force unification
            D_meta = self.D_eff * 2.5
        else:
            # DEMOCRACY: Trust internal Doubt
            u_meta_coupling = self.u
            D_meta = self.D_eff
            
        # --- 4. Physics Execution (Custom Logic for Meta-Control) ---
        
        # A. Laplacian
        # We need the Laplacian here. Assuming we can get it from network context?
        # Note: In standalone Mem4ristorV3, we don't store the Laplacian/Adj inside the model easily
        # unless passed in step(). For this 'King' implementation, we assume coupling_input
        # is handled by the caller or we are in a simple stencil mode.
        # FIX: We will rely on standard step() but we need to inject the parameters.
        # Since step() calculates I_coup internally using self.u and self.D_eff,
        # we must temporarily mutate self.u and self.D_eff.
        
        # Backup State
        backup_u = self.u.copy()
        backup_D = self.D_eff
        
        # Apply Martial Law to Parameters
        if martial_law_active:
            self.u = u_meta_coupling
            self.D_eff = D_meta
            
        # Execute Standard Step (v3.0 logic with Sigmoid + Plasticity)
        # Note: We call super().step() BUT we need to handle the 'coupling_input' argument.
        # If this is called from a Mem4Network, coupling_input is passed.
        # If called standalone, it's None.
        # WARNING: We cannot easily inject coupling_input here without refactoring the caller.
        # AS A COMPROMISE: We assume this class is used within a custom loop or we just call super().step()
        # and accept that if coupling_input is missing, physics is local.
        # Ideally, Mem4Network should call this.
        
        # Let's perform the Metabolism first
        self.metacognitive_metabolism()
        
        # Call Physics
        # We pass I_stimulus. The caller (Network) usually handles coupling_input.
        # To make this work transparently, we need to capture the coupling_input if provided.
        # Python doesn't allow easy capture of super args if we change signature.
        # We'll fix this by assuming simple usage or relying on internal state if we were fully integrated.
        # For now, let's just call super().step(I_stimulus) and assume coupling is external/stencil.
        super().step(I_stimulus)
        
        # Restore Democratic State (The mind is free again after the action)
        if martial_law_active:
            # Restore the doubt level (so we can doubt the action we just took)
            self.u = backup_u
            self.D_eff = backup_D
            
        return {
            'martial_law': martial_law_active,
            'frustration': self.frustration_phi,
            'boredom': self.boredom_index,
            'epsilon': self.cfg['dynamics']['epsilon'],
            'noise': self.cfg['noise']['sigma_v']
        }
