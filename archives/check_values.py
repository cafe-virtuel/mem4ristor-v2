from mem4ristor.core import Mem4ristorV2
import numpy as np

model = Mem4ristorV2(seed=42)
model._initialize_params(N=100)
adj = np.zeros((100, 100))
for i in range(100):
    for j in [i-1, i+1, i-10, i+10]:
        if 0 <= j < 100: adj[i, j] = 1

sol = model.solve_rk45((0, 1.0), I_stimulus=0.5, adj_matrix=adj)
h = model.calculate_entropy()
print(f"Entropy after T=1.0: {h:.4f}")
