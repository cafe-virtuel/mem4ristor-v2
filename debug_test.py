from mem4ristor.core import Mem4ristorV2
import numpy as np
import networkx as nx

def debug_rk45_stability():
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    
    adj = np.zeros((100, 100))
    # Full lattice for test
    for i in range(100):
        for j in [i-1, i+1, i-10, i+10]:
            if 0 <= j < 100: adj[i, j] = 1
            
    # Solve for 100 steps (1.0 time units if dt=0.01)
    sol = model.solve_rk45((0, 1.0), I_stimulus=0.5, adj_matrix=adj)
    
    # Entropy should be high
    h_final = model.calculate_entropy()
    print(f"h_final: {h_final}")
    assert h_final > 0.9

if __name__ == "__main__":
    debug_rk45_stability()
