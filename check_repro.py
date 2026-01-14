from mem4ristor.core import Mem4ristorV2
import numpy as np

cfg = {
    'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 4.0, 'dt': 0.1},
    'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
    'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
    'noise': {'sigma_v': 0.05}
}

model1 = Mem4ristorV2(config=cfg, seed=123)
model2 = Mem4ristorV2(config=cfg, seed=123)

for i in range(50):
    model1.step(I_stimulus=0.5)
    model2.step(I_stimulus=0.5)
    if not np.allclose(model1.v, model2.v):
        print(f"Failed at step {i}")
        print(f"Diff: {np.max(np.abs(model1.v - model2.v))}")
        break
else:
    print("Reproducibility OK")
