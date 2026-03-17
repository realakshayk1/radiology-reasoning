"""
seed.py — Global seed setter for reproducibility.
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set global seed for all common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure determinant algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
