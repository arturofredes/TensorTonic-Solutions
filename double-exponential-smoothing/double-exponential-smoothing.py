import numpy as np
def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    """
    Returns the smoothed level at every time step.
    """
    series = np.asarray(series)
    smoothed = np.zeros((series.shape))
    l_o = series[0]
    smoothed[0] = l_o
    b_o = series[1] - series[0]
    for i in range(1, len(series)):
        l = alpha * series[i] + (1 - alpha) * (l_o + b_o)
        smoothed[i] = l
        b = beta * (l - l_o) + (1 - beta) * b_o
        l_o = l
        b_o = b
    return smoothed.tolist()