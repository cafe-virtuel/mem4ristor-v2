import os
# MKL Determinism Fix (v2.9.1)
os.environ['NUMPY_MKL_CBWR'] = 'COMPATIBLE'
import numpy as np
import yaml
from typing import Dict, List, Optional
from scipy.integrate import solve_ivp

class Mem4ristorV2:
    """
    Canonical Implementation of Mem4ristor v2.9.2 (Edison Integrity Fix).
    
    Implements extended FitzHugh-Nagumo dynamics with constitutional doubt (u)
    and structural heretics for diversity preservation in neuromorphic-inspired computational models.
    
    Core Equations:
        dv/dt = v - v³/5 - w + I_ext - α·tanh(v) + η(t)
        dw/dt = ε(v + a - bw)
        du/dt = ε_u(k_u·σ_social + σ_baseline - u)
    
    Key Features:
        - Repulsive social coupling: (1-2u) filter enables active disagreement
        - Heretic units: 15% with inverted stimulus polarity
        - 1/√N scaling for consistent dynamics across network sizes
    
    Attributes:
        N (int): Number of units in network
        v (ndarray): Cognitive potential states (N,)
        w (ndarray): Recovery/inhibition states (N,)
        u (ndarray): Constitutional doubt levels (N,), clamped [0,1]
        heretic_mask (ndarray): Boolean mask identifying heretic units
        D_eff (float): Effective coupling strength = D/√N
        cfg (dict): Configuration parameters from YAML
    
    Example:
        >>> model = Mem4ristorV2(seed=42)
        >>> model._initialize_params(N=100)
        >>> adj = build_lattice_adjacency(10)
        >>> for _ in range(1000):
        ...     model.step(I_stimulus=0.0, adj_matrix=adj)
        >>> entropy = model.calculate_entropy()
        >>> print(f"Final entropy: {entropy:.4f}")
    
    References:
        See docs/preprint.tex for complete mathematical specification.
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        if config is None:
            # Try to load default config
            try:
                cfg_path = os.path.join(os.path.dirname(__file__), "../../reproduction/CONFIG_DEFAULT.yaml")
                with open(cfg_path, 'r') as f:
                    self.cfg = yaml.safe_load(f)
            except (FileNotFoundError, yaml.YAMLError):
                # Fallback to hardcoded defaults if YAML missing or invalid
                self.cfg = {
                'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
                'coupling': {'D': 0.15, 'heretic_ratio': 0.15}, # SNR Hardened
                'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
                'noise': {'sigma_v': 0.05} # SNR Hardened
            }
        else:
            self.cfg = config

        self.rng = np.random.RandomState(seed)
        self.dt = self.cfg['dynamics']['dt']
        
        # Dimensions are defined at model level or implicitly by state
        self.N = 100 # Default
        self._initialize_params()

    def _initialize_params(self, N=100, cold_start=False):
        self.N = N
        if cold_start:
            self.v = np.zeros(self.N)
            self.w = np.zeros(self.N)
        else:
            self.v = self.rng.uniform(-1.5, 1.5, self.N)
            self.w = self.rng.uniform(0.0, 1.0, self.N)
            
        self.u = np.full(self.N, self.cfg['doubt']['sigma_baseline'])
        
        # Anti-Clustering Placement (Kimi v2.9.1 - Fixed heretic_ratio=0 crash)
        heretic_ratio = self.cfg['coupling'].get('heretic_ratio', 0.15)
        
        if heretic_ratio <= 0:
            # No heretics - ablation configuration
            self.heretic_mask = np.zeros(self.N, dtype=bool)
        elif self.cfg['coupling'].get('uniform_placement', True):
            # Stratified placement: 1 heretic per block of size 1/ratio
            step = int(1.0 / heretic_ratio)
            if step < 1:
                step = 1
            heretic_ids = []
            for i in range(0, self.N, step):
                if len(heretic_ids) < int(self.N * heretic_ratio):
                    # Pick one random index in this block
                    block_end = min(i + step, self.N)
                    heretic_ids.append(self.rng.randint(i, block_end))
            self.heretic_mask = np.zeros(self.N, dtype=bool)
            self.heretic_mask[heretic_ids] = True
        else:
            # Random placement fallback
            self.heretic_mask = self.rng.rand(self.N) < heretic_ratio
            
        self.D_eff = self.cfg['coupling']['D'] / np.sqrt(self.N)

    def step(self, I_stimulus: float = 0.0, coupling_input: Optional[np.ndarray] = None) -> None:
        """
        Advance system by one time step using Euler integration.
        
        Args:
            I_stimulus: External stimulus magnitude (default: 0.0)
            coupling_input: Adjacency matrix (N, N) or pre-calculated Laplacian vector (N,).
                           If (N,N), computes: (adj @ v - v).
                           If (N,), treats as the direct Laplacian result.
        
        Returns:
            None (updates internal state in-place)
        """
        # GUARD 0: NaN detection on state variables (v2.9.3 Antigravity fix)
        # Prevents NaN propagation from corrupted state to entire network via coupling
        if np.any(np.isnan(self.v)):
            self.v = np.nan_to_num(self.v, nan=0.0)
        if np.any(np.isnan(self.w)):
            self.w = np.nan_to_num(self.w, nan=0.0)
        if np.any(np.isnan(self.u)):
            self.u = np.nan_to_num(self.u, nan=0.5)  # Default to maximum uncertainty
        
        if coupling_input is None:
            laplacian_v = np.zeros(self.N)
        elif coupling_input.ndim == 2:
            # Traditional calculation from adjacency matrix: (Avg(neighbors) - self)
            laplacian_v = coupling_input @ self.v - self.v
        else:
            # Direct Laplacian vector passed (from stencil or pre-computed matrix L)
            laplacian_v = coupling_input
        
        # GUARD 1: NaN detection in coupling (v2.9.1 Kimi fix)
        if np.any(np.isnan(laplacian_v)):
            laplacian_v = np.nan_to_num(laplacian_v, nan=0.0)
            
        sigma_social = np.abs(laplacian_v)
        
        # Noise Modeling
        # 1. Standard Gaussian Thermal Noise
        eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), self.N)
        
        # 2. RTN (Random Telegraph Noise) - Physical Hardware Signature
        # Simulates binary state jumps in memristor conductance
        if self.cfg['noise'].get('use_rtn', False):
            rtn_amp = self.cfg['noise'].get('rtn_amplitude', 0.1)
            # p_flip is the probability of a state jump per step
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            # RTN implementation: binary flips scaled by amplitude
            rtn_jumps = (self.rng.rand(self.N) < p_flip).astype(float) * rtn_amp * self.rng.choice([-1, 1], size=self.N)
            eta += rtn_jumps
            
        # Core Repulsion Kernel (v2.9.1)
        u_filter = (1.0 - 2.0 * self.u)
        I_coup = self.D_eff * u_filter * laplacian_v
        
        if np.isscalar(I_stimulus):
            I_eff = np.full(self.N, float(I_stimulus))
        else:
            I_eff = np.array(I_stimulus).flatten()
            if I_eff.size != self.N:
                raise ValueError(f"Stimulus vector size {I_eff.size} must match network size {self.N}")
        
        # GUARD 2: Clamp stimulus to prevent overflow (v2.9.1 Kimi fix)
        I_eff = np.clip(I_eff, -100.0, 100.0)
        
        # Apply heretic inversion
        I_eff[self.heretic_mask] *= -1.0
        I_ext = I_eff + I_coup
        
        # FHN Dynamics
        dv = (self.v - (self.v**3)/self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext - \
              self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)
        dw = self.cfg['dynamics']['epsilon'] * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)
        du = (self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social + \
                                               self.cfg['doubt']['sigma_baseline'] - self.u)) / self.cfg['doubt']['tau_u']
        
        self.v += dv * self.dt
        self.w += dw * self.dt
        self.u += du * self.dt
        
        # GUARD 3: Clamp v and w to prevent runaway divergence (v2.9.1 Kimi fix)
        self.v = np.clip(self.v, -100.0, 100.0)
        self.w = np.clip(self.w, -100.0, 100.0)
        self.u = np.clip(self.u, self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])

    def solve_rk45(self, t_span, I_stimulus=0.0, adj_matrix=None):
        """
        High-precision integration using RK45 for long-term stability (v2.9.1).
        """
        def combined_dynamics(t, y):
            # Split state
            N = self.N
            v = y[:N]
            w = y[N:2*N]
            u = y[2*N:]
            
            # Laplacian
            if adj_matrix is None:
                laplacian_v = np.zeros(N)
            else:
                laplacian_v = adj_matrix @ v - v
                
            sigma_social = np.abs(laplacian_v)
            
            # Doubt & Stimulus
            u_filter = (1.0 - 2.0 * u)
            I_eff = np.full(N, float(I_stimulus))
            I_eff[self.heretic_mask] *= -1.0
            I_ext = I_eff + self.D_eff * u_filter * laplacian_v
            
            # Noise (Additive white noise is tricky in IVP, we use a fixed seed per step or just mean behavior)
            # For H stability tests, we often look at deterministic attractor or low-noise limit
            eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.02), N)
            
            dv = v - (v**3)/self.cfg['dynamics']['v_cubic_divisor'] - w + I_ext - \
                 self.cfg['dynamics']['alpha'] * np.tanh(v) + eta
            dw = self.cfg['dynamics']['epsilon'] * (v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * w)
            du = (self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social + \
                                                   self.cfg['doubt']['sigma_baseline'] - u)) / self.cfg['doubt']['tau_u']
            
            return np.concatenate([dv, dw, du])

        y0 = np.concatenate([self.v, self.w, self.u])
        sol = solve_ivp(combined_dynamics, t_span, y0, method='RK45', rtol=1e-6)
        
        # Update state to last point
        y_final = sol.y[:, -1]
        self.v = y_final[:self.N]
        self.w = y_final[self.N:2*self.N]
        self.u = np.clip(y_final[2*self.N:], self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])
        return sol

    def get_states(self) -> np.ndarray:
        states = np.zeros(self.N, dtype=int)
        v = self.v
        states[v < -1.5] = 1
        states[(v >= -1.5) & (v < -0.8)] = 2
        states[(v >= -0.8) & (v <= 0.8)] = 3
        states[(v > 0.8) & (v <= 1.5)] = 4
        states[v > 1.5] = 5
        return states

    def calculate_entropy(self, bins=5, use_cognitive_bins=True) -> float:
        """
        Shannon entropy of the activation state distribution.
        Used as the primary metric for cognitive diversity.
        
        Args:
            bins: Number of bins for histogram (used if use_cognitive_bins=False)
            use_cognitive_bins: If True, uses state thresholds aligned with get_states()
                               for scientifically consistent entropy measurement.
        """
        v = self.v
        if use_cognitive_bins:
            # Aligned with get_states() thresholds: [-inf, -1.5, -0.8, 0.8, 1.5, +inf]
            # This ensures entropy measures the same cognitive states as Table 1 in preprint
            bin_edges = [-np.inf, -1.5, -0.8, 0.8, 1.5, np.inf]
            counts, _ = np.histogram(v, bins=bin_edges)
        else:
            # Legacy uniform binning (kept for backward compatibility with tests)
            counts, _ = np.histogram(v, bins=bins, range=(-5.0, 5.0))
        total = np.sum(counts)
        if total == 0: return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

class Mem4Network:
    """High-level API for Mem4ristorV2 with formal Laplacian operators."""
    def __init__(self, size: int = 10, heretic_ratio: float = 0.15, seed: int = 42,
                 adjacency_matrix: Optional[np.ndarray] = None, cold_start: bool = False,
                 boundary: str = 'periodic'):
        """
        Args:
            size: Grid side length (total units = size * size)
            heretic_ratio: Fraction of heretic units [0, 1]
            seed: Random seed for reproducibility
            adjacency_matrix: Optional explicit adjacency matrix (overrides stencil)
            cold_start: If True, initialize all units to identical state
            boundary: Boundary condition for stencil Laplacian:
                      'periodic' (toroidal wrap, default) or 'neumann' (zero-flux)
        """
        # v2.9.3 fix: Use local RNG instead of global np.random.seed
        # Global seed mutation caused non-deterministic behavior when
        # multiple Mem4Network instances were created in the same session
        self.rng = np.random.RandomState(seed)
        self.boundary = boundary
            
        self.adjacency_matrix = adjacency_matrix
        if adjacency_matrix is not None:
            self.N = adjacency_matrix.shape[0]
            self.size = int(np.sqrt(self.N))
            self.use_stencil = False
            # Formal Graph Laplacian L = D - A
            # Note: We use -L for coupling to match the repulsive/attractive sign convention
            deg = np.array(np.sum(adjacency_matrix, axis=1)).flatten()
            D = np.diag(deg)
            self.L = D - adjacency_matrix
        else:
            self.size = size
            self.N = size * size
            self.use_stencil = True
            self.L = None
            
        self.model = Mem4ristorV2(seed=seed)
        self.model.cfg['coupling']['heretic_ratio'] = heretic_ratio
        self.model._initialize_params(self.N, cold_start=cold_start)

    def get_spectral_gap(self) -> float:
        """
        Calculate the algebraic connectivity (Fiedler value).
        Measures how fast information spreads through the network.
        """
        if self.use_stencil:
            # We would need to build the sparse matrix for the stencil to get the eigenvalues
            # For now, return a placeholder or implement the periodic lattice eigenvalues
            return 0.0 
        
        from scipy.linalg import eigh
        vals = eigh(self.L, eigvals_only=True)
        # The eigenvalues are sorted, vals[0] is always ~0 for a connected graph
        # vals[1] is the spectral gap
        return vals[1] if len(vals) > 1 else 0.0

    def _calculate_laplacian_stencil(self, v):
        """
        Standard Discrete 2D Laplacian using 5-point stencil.
        L[i,j] = v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j]
        
        Boundary conditions (v2.9.3 Antigravity fix):
          - 'periodic': Toroidal wrap (default). All units have exactly 4 neighbors.
            Eliminates the border artifact where 36% of units had zero coupling.
            Physically equivalent to an infinite tiling of the lattice.
          - 'neumann': Zero-flux (dv/dn = 0). Border neighbors are reflected copies.
            Physically equivalent to an insulating boundary.
        """
        s = self.size
        v_grid = v.reshape((s, s))
        
        if self.boundary == 'periodic':
            # Periodic (toroidal) boundary conditions
            # np.roll handles the wrap-around: index -1 wraps to last, index s wraps to 0
            output = (np.roll(v_grid, 1, axis=0) + np.roll(v_grid, -1, axis=0) +
                      np.roll(v_grid, 1, axis=1) + np.roll(v_grid, -1, axis=1) -
                      4 * v_grid)
        elif self.boundary == 'neumann':
            # Neumann (zero-flux) boundary conditions
            # Pad with reflected values (mirror at edges)
            padded = np.pad(v_grid, 1, mode='edge')
            output = (padded[0:-2, 1:-1] + padded[2:, 1:-1] +
                      padded[1:-1, 0:-2] + padded[1:-1, 2:] -
                      4 * v_grid)
        else:
            raise ValueError(f"Unknown boundary condition '{self.boundary}'. Use 'periodic' or 'neumann'.")
        
        return output.flatten()

    def step(self, I_stimulus: float = 0.0):
        """Perform one simulation step."""
        if self.use_stencil:
            # Note: The sign of L determines attraction vs repulsion.
            # In social models, L*v (neighbor_avg - self) is attractive.
            # Our stencil matches this convention.
            l_v = self._calculate_laplacian_stencil(self.v)
        else:
            # Using Matrix Laplacian: - (L @ v) gives the same attractive term (neighbor_sum - deg*self)
            l_v = -(self.L @ self.v)
            
        self.model.step(I_stimulus, l_v)

    @property
    def v(self): return self.model.v
    
    def calculate_entropy(self, **kwargs): return self.model.calculate_entropy(**kwargs)
    
    def get_state_distribution(self):
        states = self.model.get_states()
        counts = np.bincount(states, minlength=6)[1:]
        return {f"bin_{i}": int(c) for i, c in enumerate(counts)}

