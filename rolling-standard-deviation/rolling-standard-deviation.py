import math
import numpy as np
def rolling_std(values: list, window_size: int) -> list:
    """
    Returns the population standard deviation of every complete window.
    """
    values = np.asarray(values)
    result = np.zeros((len(values) - window_size) + 1)
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        mu = window.mean()
        result[i] = math.sqrt(((window - mu) ** 2).mean())

    return result.tolist()
        