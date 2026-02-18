import numpy as np
from typing import Optional, Tuple

class SensoryFrontend:
    """
    Layer 0 (The Eye) - Sensory Transduction.
    
    Converts high-dimensional raw data (Images) into low-dimensional
    stimulus vectors for the Mem4ristor.
    
    Mechanism:
    - Feature Extraction: Uses localized filters (Edge/Blob detection) 
      to extract salient features from the image.
    - Projection: Maps these features to the N neurons of the Mem4ristor.
    """
    def __init__(self, output_dim: int, input_shape: Tuple[int, int] = (64, 64), seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.output_dim = output_dim
        self.input_shape = input_shape
        
        # 1. Initialize Random Filters
        self.num_filters = 16
        self.filter_size = 5
        self.filters = self.rng.normal(0, 1, (self.num_filters, self.filter_size, self.filter_size))
        
        # 2. Calculate Feature Dimension Dynamically
        # Convolve reduces dims by filter_size - 1
        conv_h = input_shape[0] - self.filter_size + 1
        conv_w = input_shape[1] - self.filter_size + 1
        
        # Pooling reduces by stride (4)
        pool_h = conv_h // 4
        pool_w = conv_w // 4
        
        feature_dim = self.num_filters * pool_h * pool_w
        
        # We use a sparse random projection to keep it efficient
        self.projection = self.rng.normal(0, 0.05, (feature_dim, self.output_dim))
        
    def _convolve(self, image: np.ndarray, filters: np.ndarray) -> np.ndarray:
        """Simple functional 2D convolution."""
        h, w = image.shape
        n_filters, fh, fw = filters.shape
        out_h = h - fh + 1
        out_w = w - fw + 1
        
        # Optimized sliding window is complex in pure numpy without stride_tricks
        # We'll do a simple loop for clarity/stability (images are small)
        output = np.zeros((n_filters, out_h, out_w))
        
        for k in range(n_filters):
            f = filters[k]
            # Flip filter for true convolution, but valid correlation is fine here
            for i in range(out_h):
                for j in range(out_w):
                    patch = image[i:i+fh, j:j+fw]
                    output[k, i, j] = np.sum(patch * f)
                    
        return output

    def _pool(self, feature_map: np.ndarray, stride: int = 4) -> np.ndarray:
        """Max pooling."""
        n, h, w = feature_map.shape
        out_h = h // stride
        out_w = w // stride
        output = np.zeros((n, out_h, out_w))
        
        for k in range(n):
            for i in range(out_h):
                for j in range(out_w):
                    grad = feature_map[k, i*stride:(i+1)*stride, j*stride:(j+1)*stride]
                    output[k, i, j] = np.max(grad)
        return output

    def perceive(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image and return the Mem4ristor stimulus.
        
        Args:
            image (np.ndarray): Grayscale image, shape (H, W). Values [0, 1].
            
        Returns:
            np.ndarray: Stimulus vector I_eff, shape (output_dim,).
        """
        if image.shape != self.input_shape:
            # Resize logic would go here, but for now strict check
            raise ValueError(f"Image shape {image.shape} != expected {self.input_shape}")
            
        # 1. Retina -> V1 (Convolution)
        # Edges and Textures
        feat_maps = self._convolve(image, self.filters)
        feat_maps = np.maximum(0, feat_maps) # ReLU
        
        # 2. V1 -> Pooling
        # Reduce dimensionality
        pooled = self._pool(feat_maps, stride=4)
        
        # 3. Flatten
        flat_features = pooled.flatten()
        
        # Ensure projection matches actual dimension
        if flat_features.shape[0] != self.projection.shape[0]:
            # Lazy fix: Adjust projection if dimensions changed slightly (padding)
            # Or just crash to be safe. We'll crash for rigorousness.
             raise ValueError(f"Feature dim {flat_features.shape[0]} != Projection dim {self.projection.shape[0]}")
            
        # 4. V1 -> Mem4ristor (Projection)
        stimulus = flat_features @ self.projection
        
        # Normalize to reasonable range for I_stim [-1.0, 1.0] suited for Mem4ristor
        # (Though Mem4ristor takes up to 100, we want delicate signals here)
        stimulus = np.tanh(stimulus) * 2.0 
        
        return stimulus

    def generate_test_pattern(self, pattern_type: str) -> np.ndarray:
        """Generates simple geometric shapes for testing."""
        img = np.zeros(self.input_shape)
        h, w = self.input_shape
        
        if pattern_type == "circle":
            Y, X = np.ogrid[:h, :w]
            center = (h//2, w//2)
            dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            img[dist <= h//3] = 1.0
            
        elif pattern_type == "square":
            start = h//4
            end = 3*h//4
            img[start:end, start:end] = 1.0
            
        elif pattern_type == "noise":
            img = self.rng.rand(*self.input_shape)
            
        return img
