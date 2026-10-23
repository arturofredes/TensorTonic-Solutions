import numpy as np
def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    if len(series) < 2:
        return

    if max_lag < 0:
        return 

    if max_lag >= len(series):
        return
        
    series = np.array(series)
    mu = np.mean(series)
    result = [1.0]
    denominator = np.sum((series - mu) ** 2)
    result = [1.0]  # lag 0
    
    if denominator == 0:
        return result +  [0.0] * max_lag



    for k in range(1, max_lag + 1):
        s1 = series[k:] - mu
        s2 = series[:-k] - mu

        r_k = np.sum(s1 * s2) / denominator
        result.append(float(r_k))

    return result

        
    
    