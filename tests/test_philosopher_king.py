import pytest
import numpy as np
import sys
import os

# Force insert at beginning to override installed packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.mem4ristor_king import Mem4ristorKing

class TestPhilosopherKing:

    def test_martial_law_trigger(self):
        """Verify that high frustration triggers Martial Law."""
        king = Mem4ristorKing()
        
        # 1. Simulate a persistent error
        # We target a state (v=10) that is far from current state (v~0)
        target = np.full(king.N, 10.0)
        
        # 2. Accumulate frustration
        # Threshold is 0.8. Gain is 0.05. Needs ~16 steps.
        
        martial_law_triggered = False
        for _ in range(30):
            status = king.step_with_governance(target_vector=target)
            if status['martial_law']:
                martial_law_triggered = True
                break
        
        assert martial_law_triggered, "Martial Law should trigger when error persists"
        assert king.frustration_phi > 0.8, "Frustration should be high"

    def test_metacognition_boredom(self):
        """Verify that low entropy (boredom) increases epsilon and sigma."""
        king = Mem4ristorKing()
        
        # 1. Force a 'Bored' state (Uniform state, Entropy = 0)
        king.v[:] = 0.0
        
        # Baseline params
        base_eps = king.cfg['dynamics']['epsilon']
        base_sig = king.cfg['noise']['sigma_v']
        
        # 2. Run for some steps
        for _ in range(50):
            # No stimulus, let it stay bored (if possible)
            # Actually, standard FHN might oscillate, creating entropy.
            # We force zero entropy in the metric calculation by making v uniform.
            # But step() updates v.
            # We cheat: We manually force v uniform after step to sustain boredom for the test.
            king.step_with_governance()
            king.v[:] = 0.0 # Suppress diversity manually to test the REACTION to boredom
        
        # 3. Check parameters
        new_eps = king.cfg['dynamics']['epsilon']
        new_sig = king.cfg['noise']['sigma_v']
        
        assert new_eps > base_eps, "Epsilon should increase when bored"
        assert new_sig > base_sig, "noise Sigma should increase when bored"
        
        print(f"Boredom Response: Epsilon {base_eps}->{new_eps:.4f}, Sigma {base_sig}->{new_sig:.4f}")

if __name__ == "__main__":
    t = TestPhilosopherKing()
    t.test_martial_law_trigger()
    t.test_metacognition_boredom()
    print("Philosopher King Tests Passed!")
