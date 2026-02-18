import pytest
import numpy as np
import sys
import os

# Add src to path
# Force insert at beginning to override installed packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mem4ristor
from mem4ristor.core import Mem4ristorV2

class TestV5Hysteresis:
    
    def test_hysteresis_configuration(self):
        """Verify V5 config is loaded correctly."""
        model = Mem4ristorV2()
        assert 'hysteresis' in model.cfg
        assert model.cfg['hysteresis']['enabled'] is True
        assert hasattr(model, 'mode_state')
        assert hasattr(model, 'time_in_state')
        
    def test_latching_mechanics(self):
        """
        Verify that the system 'latches' in the dead zone.
        Dead Zone: [0.35, 0.65] (default)
        """
        config = {
            'hysteresis': {
                'enabled': True,
                'theta_low': 0.35,
                'theta_high': 0.65,
                'fatigue_rate': 0.0 # Disable fatigue for pure hysteresis test
            }
        }
        model = Mem4ristorV2(config=config)
        
        # 1. Start at u=0.5 (Dead Zone), Init State=False (SAGE)
        # Should stay SAGE
        model.u[:] = 0.5
        model.mode_state[:] = False 
        model._update_hysteresis()
        assert not np.any(model.mode_state), "Should remain Sage in dead zone"
        
        # 2. Push above High Threshold (0.7 > 0.65)
        # Should switch to FOU
        model.u[:] = 0.7
        model._update_hysteresis()
        assert np.all(model.mode_state), "Should switch to Fou above theta_high"
        
        # 3. Drop back into Dead Zone (0.5)
        # Should STAY FOU (Latching confirmed)
        model.u[:] = 0.5
        model._update_hysteresis()
        assert np.all(model.mode_state), "Should LATCH in Fou state when returning to dead zone"
        
        # 4. Drop below Low Threshold (0.2 < 0.35)
        # Should switch back to SAGE
        model.u[:] = 0.2
        model._update_hysteresis()
        assert not np.any(model.mode_state), "Should switch back to Sage below theta_low"

    def test_watchdog_fatigue(self):
        """
        Verify that the V5.1 Watchdog relaxes thresholds over time.
        """
        config = {
            'hysteresis': {
                'enabled': True,
                'theta_low': 0.35,
                'theta_high': 0.65,
                'fatigue_rate': 0.1, # Fast fatigue
                'base_hysteresis': 0.2
            },
            'dynamics': {'dt': 1.0} # Large step for fast testing
        }
        model = Mem4ristorV2(config=config)
        
        # 1. Latch in FOU state
        model.u[:] = 0.6 # Inside dead zone
        model.mode_state[:] = True # Force start in FOU
        model.time_in_state[:] = 0.0
        
        model.u[:] = 0.45
        
        # Step 1: No fatigue yet
        model._update_hysteresis()
        assert np.all(model.mode_state), "Should stay Fou initially"
        
        # Step 2: Accumulate fatigue
        # dt=1.0, rate=0.1. Fatigue = 1 - exp(-0.1*t)
        # We need Fatigue > 0.5. -0.1*t < ln(0.5) ~ -0.69. t > 6.9 steps.
        
        for _ in range(10):
            model._update_hysteresis()
            
        # Should have switched by now
        assert not np.any(model.mode_state), "Watchdog should have forced switch to Sage"
        
        # Verify timer behavior: It should strictly be less than total time (10.0)
        # because it reset at least once.
        assert np.all(model.time_in_state < 10.0), "Timer should have reset at least once"
        
if __name__ == "__main__":
    t = TestV5Hysteresis()
    t.test_hysteresis_configuration()
    t.test_latching_mechanics()
    t.test_watchdog_fatigue()
    print("All V5 tests passed!")
