import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns the batch-mean KL divergence as a Python float.
    """
    Dkl = - 0.5 * (1 + log_var - np.multiply(mu,mu) - np.exp(log_var))
    return Dkl.sum(axis=1).mean()