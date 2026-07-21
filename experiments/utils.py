import torch
import numpy as np
import random


def set_seed(seed: int):
    """Seed Python, NumPy, and Torch random generators.

    :param seed: Seed value applied to CPU and, when available, CUDA RNGs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
