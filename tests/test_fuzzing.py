import sys
import os
import numpy as np
import pytest
import time
import random

# Ensure proper path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mem4ristor.core import Mem4ristorV2

def get_random_garbage():
    """Generates random garbage data to throw at the model."""
    garbage_types = [
        "string", 
        np.nan, 
        np.inf, 
        -np.inf, 
        None, 
        1j, # Complex number
        {"a": 1}, 
        [1, "2", 3], 
        object(),
        np.array([np.nan, np.inf]),
        np.full((10, 10), 9999999999.9), # Shape mismatch if N!=10 or N!=100
        1e308, # Max float
        -1e308
    ]
    return random.choice(garbage_types)

def test_vicious_fuzzing_inputs():
    """Run 50 iterations of input fuzzing on step()."""
    print("\n[TEST] Fuzzing step() inputs...")
    model = Mem4ristorV2(seed=123)
    model._initialize_params(N=50) 
    
    for i in range(50):
        garbage = get_random_garbage()
        try:
            # Randomly attack stimulus or coupling
            if random.choice([True, False]):
                model.step(I_stimulus=garbage)
            else:
                model.step(coupling_input=garbage)
                
            # If we get here, state must NOT be corrupted
            assert not np.any(np.isnan(model.v)), f"State corrupted by {garbage}"
            assert not np.any(np.isinf(model.v)), f"State corrupted by {garbage}"
            
        except (ValueError, TypeError):
            # Blocked! Good.
            pass
        except Exception as e:
            pytest.fail(f"CRASH: {type(e).__name__} with input {str(garbage)[:50]}: {e}")

def test_vicious_fuzzing_config():
    """Run 50 iterations of config mutation."""
    print("\n[TEST] Fuzzing configuration...")
    
    for i in range(50):
        # Create a mutated config
        bad_config = {
            'dynamics': {'dt': random.choice([0.05, 0.0, -0.1, "0.1", None])},
            'coupling': {'heretic_ratio': random.choice([0.15, 2.0, -0.5, "high"])},
            'doubt': {'tau_u': random.choice([1.0, 0.0, -1.0])}
        }
        
        try:
            # Config validation should catch bad values on init
            new_model = Mem4ristorV2(config=bad_config)
            # If init succeeds (e.g. valid values), step it
            new_model.step()
        except (ValueError, TypeError, KeyError):
            # Good, rejected bad config
            pass
        except Exception as e:
            pytest.fail(f"CONFIG CRASH: {type(e).__name__} with config {bad_config}: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
