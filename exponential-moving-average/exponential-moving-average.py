import numpy as np
def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    # Write code here
    values = np.asarray(values, dtype=float)
    EMA = np.zeros_like(values)
    EMA[0] = values[0]
    for i in range(1, values.shape[0]):
        EMA[i] = alpha * values[i] + (1 - alpha) * EMA[i - 1]

    return EMA.tolist()
        