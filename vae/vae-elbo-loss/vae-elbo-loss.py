import numpy as np

def vae_loss(x: np.ndarray, reconstruction: np.ndarray,
             mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns total_loss, reconstruction_loss, and kl_loss as Python floats.
    """
    diff = (x - reconstruction) ** 2
    L_rec = diff.sum(axis = 1).mean()
    Dkl = - 0.5 * (1 + log_var - np.multiply(mu,mu) - np.exp(log_var))
    L_kl = Dkl.sum(axis=1).mean()
    return {
        "reconstruction_loss": L_rec.astype(np.float64),
        "kl_loss": L_kl.astype(np.float64),
        "total_loss": (L_rec + L_kl).astype(np.float64)
    }