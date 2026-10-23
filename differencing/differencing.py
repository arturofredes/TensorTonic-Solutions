import numpy as np
def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    series = np.asarray(series)
    # Write code here
    for i in range(1, order + 1):
        s1 = series[1:]
        s2 = series[:-1]

        series = s1 - s2
   
    return series.tolist()