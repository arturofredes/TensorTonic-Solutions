import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    m = np.mean(x, axis = -1, keepdims=True)
    s = np.std(x, axis = -1, keepdims=True)

    return gamma * (x - m) / np.sqrt(s ** 2 + eps) + beta
    