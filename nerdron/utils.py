import numpy as np
from typing import Union, Optional

class GPU:
    """Utility class for GPU computations."""
    def __init__(self):
        self.use_gpu = False
        try:
            import cupy as cp
            self.cp = cp
            self.use_gpu = True
        except ImportError:
            self.cp = None
    
    def array(self, arr: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Convert array to GPU if available."""
        if self.use_gpu and not isinstance(arr, self.cp.ndarray):
            return self.cp.array(arr)
        return arr
    
    def to_cpu(self, arr: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Convert array to CPU."""
        if self.use_gpu and isinstance(arr, self.cp.ndarray):
            return self.cp.asnumpy(arr)
        return arr
