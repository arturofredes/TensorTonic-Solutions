import numpy as np
def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    # Write code here
    values = np.asarray(values)
    weights = np.asarray(weights)

    k = weights.shape[0]
    r = values.shape[0] - k + 1
    result = np.zeros((r))

    denominator = weights.sum()
    for i in range(r):
        result[i] = values[i:i + k] @ weights / denominator
        
    return result.tolist()