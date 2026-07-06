import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)

    if x.shape != p.shape:
        return None

    if not np.allclose(1.0, np.sum(p), atol=1e-06):
        raise ValueError("Invalid probabilities, should sum 1")

    xp = np.dot(x,p)
    return np.sum(xp)