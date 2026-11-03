import numpy as np
def seasonal_average(series: list, period: int) -> list:
    """
    Returns the average for each position in the seasonal cycle.
    """
    # Write code here
    series = np.asarray(series)
    result = np.zeros((period))
    for i in range(period):
        result[i] = series[i::period].mean()

    return result.tolist()