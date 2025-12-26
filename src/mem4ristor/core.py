import numpy as np
import yaml
import os
from typing import Dict, List, Optional

class Mem4ristorV2:
    """
    Canonical Implementation of Mem4ristor v2.0.4.1 (Nuclear Certified).
    Unified vectorized engine for stability and performance.
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        if config is None:
            # Try to load default config
            try:
                cfg_path = os.path.join(os.path.dirname(__file__), "../../reproduction/CONFIG_DEFAULT.yaml")
                with open(cfg_path, 'r') as f:
                    self.cfg = yaml.safe_load(f)
            except:
                # Fallback to hardcoded defaults if YAML mission
                self.cfg = {
                    'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.1},
                    'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
                    'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0]},
                    'noise': {'sigma_v': 0.05}
                }
        else:
            self.cfg = config

        self.rng = np.random.RandomState(seed)
        self.dt = self.cfg['dynamics']['dt']
        
        # Dimensions are defined at model level or implicitly by state
        self.N = 100 # Default
        self._initialize_params()

    def _initialize_params(self, N=100):
        self.N = N
        self.v = self.rng.uniform(-1.5, 1.5, self.N)
        self.w = self.rng.uniform(0.0, 1.0, self.N)
        self.u = np.full(self.N, self.cfg['doubt']['sigma_baseline'])
        self.heretic_mask = self.rng.rand(self.N) < self.cfg['coupling']['heretic_ratio']
        self.D_eff = self.cfg['coupling']['D'] / np.sqrt(self.N)

    def step(self, I_stimulus: float = 0.0, adj_matrix: Optional[np.ndarray] = None):
        if adj_matrix is None:
            # Default to all-to-all if no matrix provided (simplified)
            laplacian_v = np.mean(self.v) - self.v
        else:
            laplacian_v = adj_matrix @ self.v - self.v
            
        sigma_social = np.abs(laplacian_v)
        eta = self.rng.normal(0, self.cfg['noise']['sigma_v'], self.N)
        
        # Core Repulsion Kernel (v2.0.4.1)
        u_filter = (1.0 - 2.0 * self.u)
        I_coup = self.D_eff * u_filter * laplacian_v
        
        I_eff = np.full(self.N, float(I_stimulus))
        I_eff[self.heretic_mask] *= -1.0
        I_ext = I_eff + I_coup
        
        # FHN Dynamics
        dv = (self.v - (self.v**3)/self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext - \
              self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)
        dw = self.cfg['dynamics']['epsilon'] * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)
        du = self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social + \
                                               self.cfg['doubt']['sigma_baseline'] - self.u)
        
        self.v += dv * self.dt
        self.w += dw * self.dt
        self.u += du * self.dt
        self.u = np.clip(self.u, self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])

    def get_states(self) -> np.ndarray:
        states = np.zeros(self.N, dtype=int)
        v = self.v
        states[v < -1.5] = 1
        states[(v >= -1.5) & (v < -0.8)] = 2
        states[(v >= -0.8) & (v <= 0.8)] = 3
        states[(v > 0.8) & (v <= 1.5)] = 4
        states[v > 1.5] = 5
        return states

    def calculate_entropy(self) -> float:
        states = self.get_states()
        counts = np.bincount(states, minlength=6)[1:]
        probs = counts / self.N
        probs = probs[probs > 0]
        if len(probs) <= 1: return 0.0
        return -np.sum(probs * np.log2(probs))

class Mem4Network:
    """High-level Grid-friendly API for Mem4ristorV2."""
    def __init__(self, size: int, heretic_ratio: float = 0.15, seed: int = 42):
        self.size = size
        self.N = size * size
        self.model = Mem4ristorV2(seed=seed)
        self.model._initialize_params(self.N)
        self.adj = self._build_lattice(size)

    def _build_lattice(self, size):
        adj = np.zeros((self.N, self.N))
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                neighbors = []
                if i > 0: neighbors.append((i-1) * size + j)
                if i < size - 1: neighbors.append((i+1) * size + j)
                if j > 0: neighbors.append(i * size + (j-1))
                if j < size - 1: neighbors.append(i * size + (j+1))
                for n in neighbors:
                    adj[idx, n] = 1.0 / len(neighbors)
        return adj

    def step(self, I_stimulus: float = 0.0):
        self.model.step(I_stimulus, self.adj)

    @property
    def v(self): return self.model.v
    
    def calculate_entropy(self): return self.model.calculate_entropy()
    
    def get_state_distribution(self):
        states = self.model.get_states()
        counts = np.bincount(states, minlength=6)[1:]
        return {f"bin_{i}": int(c) for i, c in enumerate(counts)}
