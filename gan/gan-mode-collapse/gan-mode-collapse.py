import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Returns diversity_score and is_collapsed in a dictionary.
    """
    sigma = np.std(generated_samples, axis = 0)
    s = sigma.mean()
    is_collapsed = True if s < threshold else False
    return {"diversity_score": s.mean(),
           "is_collapsed": is_collapsed}