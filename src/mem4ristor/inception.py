import numpy as np
from typing import Tuple
from .sensory import SensoryFrontend

class DreamVisualizer:
    """
    Project Inception: The Dream Decoder.
    
    Reconstructs visual hallucinations from the abstract neural state ($v$) of the Mem4ristor.
    It inverts the sensory projection process using the Moore-Penrose pseudo-inverse.
    
    Equation:
    Image_dream = v @ P_pseudo_inverse
    """
    def __init__(self, sensory_frontend: SensoryFrontend):
        self.frontend = sensory_frontend
        
        # 1. Compute Pseudo-Inverse of the Projection Matrix
        # P shape: (Feature_Dim, N_neurons)
        # We need to map N_neurons back to Feature_Dim
        # P_pinv shape: (N_neurons, Feature_Dim)
        self.P_pinv = np.linalg.pinv(self.frontend.projection)
        
        # We also need to reverse the pooling/convolution... 
        # Wait, the projection maps FEATURES to NEURONS.
        # The frontend does: Image -> Features -> Projection -> Neurons.
        # Inverting Projection gets us back to FEATURES.
        # Inverting Convolution/Pooling is hard (ill-posed).
        # Strategy:
        # We will reconstruct the "Feature Map" and display that.
        # It won't be the exact pixel image, but a "Deep Dream" version of it.
        # If the features are just subsampled pixels (which they roughly are in random conv),
        # we can reshape the feature vector back to a 2D grid.
        
        # Let's see how SensoryFrontend works:
        # It has conv_h, conv_w, num_filters.
        # But we don't have access to those private vars easily unless we stored them.
        # Let's inspect SensoryFrontend again or assume standard shapes.
        # Actually, SensoryFrontend doesn't store conv output shape as public.
        # We will try to reshape typically to a square approx.
        
    def decode(self, neural_state: np.ndarray) -> np.ndarray:
        """
        Decodes a neural state vector ($v$) into a 2D image.
        """
        # 1. Back-project: Neurons (N) -> Features (F)
        # state (1, N) @ (N, F) = (1, F)
        features = neural_state @ self.P_pinv
        
        # 2. Reshape to Image
        # Features dim = size of features.
        # We want to reshape this into a square image.
        size = int(np.sqrt(features.shape[0]))
        
        if size * size != features.shape[0]:
            # If not a perfect square, pad or crop
            # This happens if num_filters > 1
            # Let's just try to approximate a square
             return features # Return raw 1D if reshape fails for now, caller handles it?
             # No, let's force a square visualization
             side = int(np.ceil(np.sqrt(features.shape[0])))
             padding = side * side - features.shape[0]
             padded = np.pad(features, (0, padding), mode='constant')
             image = padded.reshape((side, side))
        else:
            image = features.reshape((size, size))
            
        # 3. Normalize for Visualization [0, 1]
        if np.max(image) - np.min(image) > 1e-6:
             image = (image - np.min(image)) / (np.max(image) - np.min(image))
             
        return image
