import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.asarray(x)
    x = np.sort(x)

    if len(x) < 1:
        return None

    
    result = np.zeros(len(q), dtype=float)
    for i, qi in enumerate(q):
        if (qi < 0) or (qi > 100):
            return None
        result[i] = np.percentile(x, qi, method = "linear")

    return result
    