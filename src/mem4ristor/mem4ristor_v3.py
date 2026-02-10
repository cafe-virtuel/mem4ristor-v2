import numpy as np
from typing import Optional, Dict

try:
    # Package import (when installed or PYTHONPATH set)
    from mem4ristor.core import Mem4ristorV2, Mem4Network
except ImportError:
    # Fallback for direct script execution
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core import Mem4ristorV2, Mem4Network

class Mem4ristorV3(Mem4ristorV2):
    """
    Mem4ristor v3.0 (Generative) - Architected by GLM-4.7.
    
    Major Upgrades:
    1. Levitating Sigmoid Transfer Function (Math Patch):
       Replaces the 'Dead Zone' at u=0.5 with a smooth, non-zero phase inversion.
       f(u) = tanh(pi * (u - 0.5)) + delta
       
    2. Inhibition Plasticity (Innovation Engine):
       The recovery variable 'w' now learns from social stress.
       If a unit is DISSIDENT (u > 0.5) and under PRESSURE (sigma_social > 0),
       it strengthens its inhibition memory to "lock in" the alternative view.
       
    3. Structural Learning:
       Dissidence is no longer just a transient state, but leaves a permanent trace
       in the synaptic weight 'w'.
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        super().__init__(config, seed)
        
        # v3.0 Specific Configuration (Defaults if not in config)
        dynamics_cfg = self.cfg.get('dynamics', {})
        self.lambda_learn = dynamics_cfg.get('lambda_learn', 0.05)      # Plasticity learning rate
        self.tau_plasticity = dynamics_cfg.get('tau_plasticity', 1000)  # Forgetting time constant
        self.w_saturation = dynamics_cfg.get('w_saturation', 2.0)       # Maximum plasticity range
        
        # Tuning for Levitating Sigmoid
        self.sigmoid_steepness = np.pi
        self.social_leakage = 0.05 # The 'delta' ensuring never-zero coupling

    def step(self, I_stimulus: float = 0.0, coupling_input: Optional[np.ndarray] = None) -> None:
        """
        Advance system by one time step using Euler integration.
        Overrides v2.6 step to implement Levitating Sigmoid and Plasticity.
        """
        # --- 1. Compute Laplacian & Social Magnitude ---
        if coupling_input is None:
            laplacian_v = np.zeros(self.N)
        elif coupling_input.ndim == 2:
            laplacian_v = coupling_input @ self.v - self.v
        else:
            laplacian_v = coupling_input
            
        # GUARD: NaN detection
        if np.any(np.isnan(laplacian_v)):
            laplacian_v = np.nan_to_num(laplacian_v, nan=0.0)
            
        sigma_social = np.abs(laplacian_v)
        
        # --- 2. Noise & External Input ---
        eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), self.N)
        
        # stim vector
        if np.isscalar(I_stimulus):
            I_eff = np.full(self.N, float(I_stimulus))
        else:
            I_eff = np.array(I_stimulus).flatten()
            
        I_eff = np.clip(I_eff, -100.0, 100.0)
        I_eff[self.heretic_mask] *= -1.0
        
        # --- 3. v3.0 CORE: Levitating Sigmoid Coupling ---
        # f(u) = tanh(pi * (u - 0.5)) + delta
        # If u=0 (Target +), f ~ -1 (Attractive normalized? wait, v2.6 logic inverted)
        # v2.6 Logic: (1-2u) => u=0 -> +1 (Attraction), u=1 -> -1 (Repulsion)
        #
        # GLM-4.7 Logic: "Inverser la phase".
        # tanh(pi*(u-0.5)) -> u=0 => tanh(-pi/2) ~ -0.9. u=1 => +0.9.
        # This REVERSES the v2.6 convention unless we flip signs.
        # v2.6: u=0 means Trust (Attraction). u=1 means Doubt (Repulsion).
        # We want u=0 -> Positive Coupling (+).
        # We want u=1 -> Negative Coupling (-).
        #
        # Let's check tanh(pi*(u-0.5)).
        # u=0 -> tanh(-1.57) = -0.92. (Negative). This is wrong for v2.6 semantics.
        # We need NEGATIVE tanh or FLIPPED input.
        # Let's use: tanh(pi * (0.5 - u)).
        # u=0 -> tanh(0.5pi) = +0.92 (Attraction).
        # u=0.5 -> 0.
        # u=1 -> -0.92 (Repulsion).
        # Correct formula for v3.0 to match v2.6 semantics:
        
        u_centered = 0.5 - self.u 
        u_filter = np.tanh(self.sigmoid_steepness * u_centered) + self.social_leakage
        
        # Note: social_leakage adds a constant positive bias (Leakage towards empathy/attraction).
        
        I_coup = self.D_eff * u_filter * laplacian_v
        I_ext = I_eff + I_coup
        
        # --- 4. v3.0 MOTOR: Inhibition Plasticity ---
        # Terms:
        # lambda * sigma_social : Learn from stress
        # I(u > 0.5) : Only if doubting (Anti-conformist Memory)
        # Soft Saturation : (1 - (w/sat)^2)
        # Decay : -(w/tau)
        
        # Innovation Mask: Units that are actively doubting/resisting
        innovation_mask = (self.u > 0.5).astype(float)
        
        plasticity_drive = self.lambda_learn * sigma_social * innovation_mask
        
        # Soft Saturation (Logistic-like damping)
        # Avoid div by zero if w_saturation is small, though defaults to 2.0
        w_ratio = self.w / self.w_saturation
        saturation_factor = 1.0 - (w_ratio**2)
        saturation_factor = np.clip(saturation_factor, 0.0, 1.0)
        
        # Plasticity Decay (Forgetting curve) to ensure long-term stability
        plasticity_decay = self.w / self.tau_plasticity
        
        dw_learning = (plasticity_drive * saturation_factor) - plasticity_decay
        
        # Standard FHN dynamics
        v_term = (self.v - (self.v**3)/self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext - \
                  self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)
        
        w_term = self.cfg['dynamics']['epsilon'] * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)
        
        # FHN u dynamics (Standard v2.6)
        # Note: We use the v2.6 implementation (div by tau_u) as validated in the audit
        du_term = (self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social + \
                                                     self.cfg['doubt']['sigma_baseline'] - self.u)) / self.cfg['doubt']['tau_u']
        
        # --- 5. Integration ---
        self.v += v_term * self.dt
        self.w += (w_term + dw_learning) * self.dt # Inject Learning here
        self.u += du_term * self.dt
        
        # Clamps
        self.v = np.clip(self.v, -100.0, 100.0)
        self.w = np.clip(self.w, -100.0, 100.0)
        self.u = np.clip(self.u, self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])
