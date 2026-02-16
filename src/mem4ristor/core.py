import os
# MKL Determinism Fix (v2.9.1)
os.environ['NUMPY_MKL_CBWR'] = 'COMPATIBLE'
import numpy as np
import yaml
from typing import Dict, List, Optional
from scipy.integrate import solve_ivp


class Mem4ristorV3:
    """
    Canonical Implementation of Mem4ristor v3.0.0.

    Implements extended FitzHugh-Nagumo dynamics with constitutional doubt (u),
    structural heretics, Levitating Sigmoid coupling, and inhibition plasticity
    for diversity preservation in neuromorphic-inspired computational models.

    Core Equations:
        dv/dt = v - v³/5 - w + I_ext - α·tanh(v) + η(t)
        dw/dt = ε(v + a - bw) + dw_plasticity
        du/dt = ε_u(k_u·σ_social + σ_baseline - u) / τ_u

    Coupling Kernel (Levitating Sigmoid):
        f(u) = tanh(π(0.5 - u)) + δ
        Eliminates the dead zone at u=0.5 where linear (1-2u) made coupling
        noise-dominated. The leakage term δ ensures non-zero coupling everywhere.

    Inhibition Plasticity:
        When a unit is dissident (u > 0.5) and under social pressure,
        its recovery variable w accumulates structural memory of dissidence.
        dw_learn = λ · σ_social · I(u>0.5) · (1 - (w/w_sat)²) - w/τ_plasticity

    Key Features:
        - Smooth repulsive coupling via Levitating Sigmoid
        - Heretic units: 15% with inverted stimulus polarity (empirical threshold)
        - 1/√N scaling for consistent dynamics across network sizes
        - Structural memory of dissidence via inhibition plasticity

    Migration Note (v2.9.3 → v3.0.0):
        This class replaces Mem4ristorV2. The linear kernel (1-2u) is replaced
        by tanh(π(0.5-u)) + δ. All security guards from V2 are preserved.
        See CHANGELOG_V3.md for complete migration details.

    Attributes:
        N (int): Number of units in network
        v (ndarray): Cognitive potential states (N,)
        w (ndarray): Recovery/inhibition states (N,)
        u (ndarray): Constitutional doubt levels (N,), clamped [0,1]
        heretic_mask (ndarray): Boolean mask identifying heretic units
        D_eff (float): Effective coupling strength = D/√N
        cfg (dict): Configuration parameters from YAML

    Example:
        >>> model = Mem4ristorV3(seed=42)
        >>> model._initialize_params(N=100)
        >>> for _ in range(1000):
        ...     model.step(I_stimulus=0.0)
        >>> entropy = model.calculate_entropy()
        >>> print(f"Final entropy: {entropy:.4f}")

    References:
        See docs/preprint.tex for complete mathematical specification.
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        # Default Configuration
        default_cfg = {
            'dynamics': {
                'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                'v_cubic_divisor': 5.0, 'dt': 0.05,
                'lambda_learn': 0.05, 'tau_plasticity': 1000, 'w_saturation': 2.0
            },
            'coupling': {'D': 0.15, 'heretic_ratio': 0.15, 'uniform_placement': True},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
            'noise': {'sigma_v': 0.05, 'use_rtn': False, 'rtn_amplitude': 0.1, 'rtn_p_flip': 0.01}
        }

        # 1. Config Hardening: Deep Merge
        if config is None:
            try:
                cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                with open(cfg_path, 'r') as f:
                    file_cfg = yaml.safe_load(f)
                    self.cfg = self._deep_merge(default_cfg, file_cfg)
            except (FileNotFoundError, yaml.YAMLError):
                self.cfg = default_cfg
        else:
            self.cfg = self._deep_merge(default_cfg, config)

        self.rng = np.random.RandomState(seed)
        self.dt = self.cfg['dynamics']['dt']

        # 2. Parameter Validation
        self._validate_config()

        # 3. V3 Plasticity Parameters
        self.lambda_learn = self.cfg['dynamics'].get('lambda_learn', 0.05)
        self.tau_plasticity = self.cfg['dynamics'].get('tau_plasticity', 1000)
        self.w_saturation = self.cfg['dynamics'].get('w_saturation', 2.0)

        # 4. Levitating Sigmoid Parameters
        self.sigmoid_steepness = np.pi
        self.social_leakage = 0.05  # δ: ensures non-zero coupling everywhere

        # Dimensions are defined at model level or implicitly by state
        self.N = 100  # Default
        self._initialize_params()

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursive merge of dictionaries to prevent missing keys."""
        result = base.copy()
        for key, value in update.items():
            if key in result:
                # GUARD: Type Confusion (Round 2 Fix)
                if isinstance(result[key], dict):
                    if not isinstance(value, dict):
                        raise TypeError(f"Config key '{key}' expects dict, got {type(value).__name__}")
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def _validate_config(self):
        """Ensure critical parameters are safe."""
        if self.cfg['dynamics']['v_cubic_divisor'] <= 1e-9:
            raise ValueError("Configuration Error: 'v_cubic_divisor' must be > 0 to prevent division by zero.")
        if self.cfg['doubt']['tau_u'] <= 1e-9:
            raise ValueError("Configuration Error: 'tau_u' must be > 0.")
        if self.dt <= 0:
            raise ValueError("Configuration Error: 'dt' must be positive.")

        D = self.cfg['coupling'].get('D', 0.15)
        if not np.isfinite(D):
            raise ValueError("Configuration Error: 'D' must be finite.")

        if self.cfg['noise'].get('use_rtn', False):
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            if not (0.0 <= p_flip <= 1.0):
                raise ValueError(f"Configuration Error: 'rtn_p_flip' must be in [0, 1], got {p_flip}")

        heretic_ratio = self.cfg['coupling'].get('heretic_ratio', 0.15)
        if not (0.0 <= heretic_ratio <= 1.0):
            raise ValueError(f"Configuration Error: 'heretic_ratio' must be in [0, 1], got {heretic_ratio}")

    def _initialize_params(self, N=100, cold_start=False):
        # Size Validation (DoS Fix)
        if N <= 0:
            raise ValueError(f"Network size N must be positive, got {N}.")
        if N > 10_000_000:
            raise ValueError(f"Network size N={N} exceeds maximum allowed (10M) to prevent DoS.")

        self.N = N
        if cold_start:
            self.v = np.zeros(self.N)
            self.w = np.zeros(self.N)
        else:
            self.v = self.rng.uniform(-1.5, 1.5, self.N)
            self.w = self.rng.uniform(0.0, 1.0, self.N)

        self.u = np.full(self.N, self.cfg['doubt']['sigma_baseline'])

        # Anti-Clustering Placement
        heretic_ratio = self.cfg['coupling'].get('heretic_ratio', 0.15)

        if heretic_ratio <= 0:
            self.heretic_mask = np.zeros(self.N, dtype=bool)
        elif self.cfg['coupling'].get('uniform_placement', True):
            step = int(1.0 / heretic_ratio)
            if step < 1:
                step = 1
            heretic_ids = []
            for i in range(0, self.N, step):
                if len(heretic_ids) < int(self.N * heretic_ratio):
                    block_end = min(i + step, self.N)
                    heretic_ids.append(self.rng.randint(i, block_end))
            self.heretic_mask = np.zeros(self.N, dtype=bool)
            self.heretic_mask[heretic_ids] = True
        else:
            self.heretic_mask = self.rng.rand(self.N) < heretic_ratio

        self.D_eff = self.cfg['coupling']['D'] / np.sqrt(self.N)

    def step(self, I_stimulus: float = 0.0, coupling_input: Optional[np.ndarray] = None) -> None:
        """
        Advance system by one time step using Euler integration.

        Implements Levitating Sigmoid coupling and inhibition plasticity.

        Args:
            I_stimulus: External stimulus magnitude (default: 0.0)
            coupling_input: Adjacency matrix (N, N) or pre-calculated Laplacian vector (N,).
                           If (N,N), computes: (adj @ v - v).
                           If (N,), treats as the direct Laplacian result.

        Returns:
            None (updates internal state in-place)
        """
        # GUARD: Deterministic Input (Round 3 Fix - Chaos Injection)
        if hasattr(I_stimulus, '__float__') and not isinstance(I_stimulus, (int, float, np.number, np.ndarray)):
            raise TypeError("Stimulus must be a numeric constant (int/float/array) to ensure reproducibility.")

        # GUARD 0: NaN/Inf detection on state variables
        if np.any(~np.isfinite(self.v)):
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(self.w)):
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(self.u)):
            self.u = np.nan_to_num(self.u, nan=0.5, posinf=0.5, neginf=0.5)

        # GUARD: Coupling Input Sanitization
        if coupling_input is not None:
            try:
                coupling_input = np.array(coupling_input, dtype=float)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"Invalid coupling input: {coupling_input}. Must be numeric.")

        if coupling_input is None:
            laplacian_v = np.zeros(self.N)
        elif coupling_input.ndim == 2:
            laplacian_v = coupling_input @ self.v - self.v
        else:
            laplacian_v = coupling_input

        # GUARD 1: NaN/Inf detection in coupling
        if np.any(np.isnan(laplacian_v)) or np.any(np.isinf(laplacian_v)):
            laplacian_v = np.nan_to_num(laplacian_v, nan=0.0, posinf=1.0, neginf=-1.0)

        sigma_social = np.abs(laplacian_v)

        # Noise Modeling
        eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), self.N)

        # RTN (Random Telegraph Noise) - Physical Hardware Signature
        if self.cfg['noise'].get('use_rtn', False):
            rtn_amp = self.cfg['noise'].get('rtn_amplitude', 0.1)
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            rtn_jumps = (self.rng.rand(self.N) < p_flip).astype(float) * rtn_amp * self.rng.choice([-1, 1], size=self.N)
            eta += rtn_jumps

        # V3 CORE: Levitating Sigmoid Coupling
        # f(u) = tanh(π(0.5 - u)) + δ
        # u=0 → tanh(π/2) ≈ +0.92 + δ (attraction)
        # u=0.5 → tanh(0) = 0 + δ (weak attraction from leakage)
        # u=1 → tanh(-π/2) ≈ -0.92 + δ (repulsion)
        u_centered = 0.5 - self.u
        u_filter = np.tanh(self.sigmoid_steepness * u_centered) + self.social_leakage

        I_coup = self.D_eff * u_filter * laplacian_v

        # Input Sanitization
        try:
            stim_arr = np.array(I_stimulus, dtype=float)
            if stim_arr.ndim == 0:
                I_eff = np.full(self.N, float(stim_arr))
            else:
                I_eff = stim_arr.flatten()
                if I_eff.size != self.N:
                    raise ValueError(f"Stimulus vector size {I_eff.size} must match network size {self.N}")
        except (ValueError, TypeError, AttributeError):
            raise ValueError(f"Invalid stimulus input: {I_stimulus}. Must be numeric.")

        if np.any(np.isnan(I_eff)):
            I_eff = np.nan_to_num(I_eff, nan=0.0)

        # GUARD 2: Clamp stimulus to prevent overflow
        I_eff = np.clip(I_eff, -100.0, 100.0)

        # Apply heretic inversion
        I_eff[self.heretic_mask] *= -1.0
        I_ext = I_eff + I_coup

        # V3 MOTOR: Inhibition Plasticity
        # Units that are actively doubting accumulate structural memory
        innovation_mask = (self.u > 0.5).astype(float)
        plasticity_drive = self.lambda_learn * sigma_social * innovation_mask

        w_ratio = self.w / self.w_saturation
        saturation_factor = np.clip(1.0 - (w_ratio**2), 0.0, 1.0)
        plasticity_decay = self.w / self.tau_plasticity

        dw_learning = (plasticity_drive * saturation_factor) - plasticity_decay

        # FHN Dynamics
        dv = (self.v - (self.v**3) / self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext -
              self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)
        dw = self.cfg['dynamics']['epsilon'] * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)
        du = (self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social +
              self.cfg['doubt']['sigma_baseline'] - self.u)) / self.cfg['doubt']['tau_u']

        self.v += dv * self.dt
        self.w += (dw + dw_learning) * self.dt
        self.u += du * self.dt

        # GUARD 3: Clamp to prevent runaway divergence
        self.v = np.clip(self.v, -100.0, 100.0)
        self.w = np.clip(self.w, -100.0, 100.0)
        self.u = np.clip(self.u, self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])

    def solve_rk45(self, t_span, I_stimulus=0.0, adj_matrix=None):
        """
        High-precision integration using RK45 for long-term stability.

        Uses the Levitating Sigmoid kernel and plasticity in continuous-time.
        """
        # GUARD: Solver Shape Mismatch
        if adj_matrix is not None and adj_matrix.shape != (self.N, self.N):
            raise ValueError(f"Shape mismatch: adj_matrix must be ({self.N}, {self.N}), got {adj_matrix.shape}")

        # GUARD: Time Span Validation
        if t_span[1] <= t_span[0]:
            raise ValueError(f"Invalid time span {t_span}: End time must be greater than start time.")
        if t_span[0] < 0 or t_span[1] < 0:
            raise ValueError(f"Invalid time span {t_span}: Time must be non-negative.")

        # GUARD: Singularity Prevention (Stiffness)
        duration = t_span[1] - t_span[0]
        max_step = min(0.1, duration / 10.0) if duration > 0 else 0.1

        def combined_dynamics(t, y):
            N = self.N
            v = y[:N]
            w = y[N:2*N]
            u = y[2*N:]

            if adj_matrix is None:
                laplacian_v = np.zeros(N)
            else:
                laplacian_v = adj_matrix @ v - v

            sigma_social = np.abs(laplacian_v)

            # V3: Levitating Sigmoid kernel
            u_centered = 0.5 - u
            u_filter = np.tanh(self.sigmoid_steepness * u_centered) + self.social_leakage

            I_eff = np.full(N, float(I_stimulus))
            I_eff[self.heretic_mask] *= -1.0
            I_ext = I_eff + self.D_eff * u_filter * laplacian_v

            eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), N)

            dv = v - (v**3) / self.cfg['dynamics']['v_cubic_divisor'] - w + I_ext - \
                 self.cfg['dynamics']['alpha'] * np.tanh(v) + eta
            dw_fhn = self.cfg['dynamics']['epsilon'] * (v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * w)

            # V3: Plasticity in continuous-time
            innovation_mask = (u > 0.5).astype(float)
            plasticity_drive = self.lambda_learn * sigma_social * innovation_mask
            w_ratio = w / self.w_saturation
            saturation_factor = np.clip(1.0 - (w_ratio**2), 0.0, 1.0)
            plasticity_decay = w / self.tau_plasticity
            dw_learn = (plasticity_drive * saturation_factor) - plasticity_decay

            dw = dw_fhn + dw_learn
            du = (self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social +
                  self.cfg['doubt']['sigma_baseline'] - u)) / self.cfg['doubt']['tau_u']

            return np.concatenate([dv, dw, du])

        y0 = np.concatenate([self.v, self.w, self.u])
        sol = solve_ivp(combined_dynamics, t_span, y0, method='RK45', rtol=1e-6, max_step=max_step)

        # GUARD: Stiff System Detection
        if sol.nfev > 50000:
            import warnings
            warnings.warn(f"High computational cost ({sol.nfev} steps). System might be stiff. Consider checking parameters.")

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
        # GUARD: Entropy Calculation Hardening
        try:
            if not use_cognitive_bins:
                bins = int(bins)
                if bins <= 0:
                    raise ValueError
                if bins > 1_000_000:
                    bins = 1_000_000
        except (ValueError, TypeError):
            bins = 5

        v = self.v
        if use_cognitive_bins:
            bin_edges = [-np.inf, -1.5, -0.8, 0.8, 1.5, np.inf]
            counts, _ = np.histogram(v, bins=bin_edges)
        else:
            counts, _ = np.histogram(v, bins=bins, range=(-5.0, 5.0))
        total = np.sum(counts)
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))


# Backward compatibility alias
Mem4ristorV2 = Mem4ristorV3


class Mem4Network:
    """High-level API for Mem4ristorV3 with formal Laplacian operators."""
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
        self.rng = np.random.RandomState(seed)
        self.boundary = boundary

        self.adjacency_matrix = adjacency_matrix
        if adjacency_matrix is not None:
            # GUARD: Matrix Sanitization
            if np.any(np.isnan(adjacency_matrix)) or np.any(np.isinf(adjacency_matrix)):
                raise ValueError("Adjacency matrix contains NaN or Inf.")

            self.N = adjacency_matrix.shape[0]
            self.size = int(np.sqrt(self.N))
            self.use_stencil = False
            deg = np.array(np.sum(adjacency_matrix, axis=1)).flatten()
            D = np.diag(deg)
            self.L = D - adjacency_matrix
        else:
            self.size = size
            self.N = size * size
            self.use_stencil = True
            self.L = None

        self.model = Mem4ristorV3(seed=seed)
        self.model.cfg['coupling']['heretic_ratio'] = heretic_ratio
        self.model._initialize_params(self.N, cold_start=cold_start)

    def get_spectral_gap(self) -> float:
        """
        Calculate the algebraic connectivity (Fiedler value).
        Measures how fast information spreads through the network.
        """
        if self.use_stencil:
            return 0.0

        # GUARD: Topological Consistency
        if not np.allclose(self.L, self.L.T, atol=1e-10):
            raise ValueError("Spectral gap requires symmetric Laplacian (undirected graph).")

        from scipy.linalg import eigh
        vals = eigh(self.L, eigvals_only=True)
        return vals[1] if len(vals) > 1 else 0.0

    def _calculate_laplacian_stencil(self, v):
        """
        Standard Discrete 2D Laplacian using 5-point stencil.
        L[i,j] = v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j]

        Boundary conditions:
          - 'periodic': Toroidal wrap (default). All units have exactly 4 neighbors.
          - 'neumann': Zero-flux (dv/dn = 0). Border neighbors are reflected copies.
        """
        s = self.size
        v_grid = v.reshape((s, s))

        if self.boundary == 'periodic':
            output = (np.roll(v_grid, 1, axis=0) + np.roll(v_grid, -1, axis=0) +
                      np.roll(v_grid, 1, axis=1) + np.roll(v_grid, -1, axis=1) -
                      4 * v_grid)
        elif self.boundary == 'neumann':
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
            l_v = self._calculate_laplacian_stencil(self.v)
        else:
            l_v = -(self.L @ self.v)

        self.model.step(I_stimulus, l_v)

    @property
    def v(self): return self.model.v

    def calculate_entropy(self, **kwargs): return self.model.calculate_entropy(**kwargs)

    def get_state_distribution(self):
        states = self.model.get_states()
        counts = np.bincount(states, minlength=6)[1:]
        return {f"bin_{i}": int(c) for i, c in enumerate(counts)}
