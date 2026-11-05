import numpy as np
def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    # Write code here
    lags = np.asarray(lags)
    max_lag = np.max(lags)
    series = np.asarray(series)
    
    n = series.shape[0]
    l = lags.shape[0]
    m = n - max_lag
    result = np.zeros((m,l))
    
    for i, lag in enumerate(lags):
        result[:, i] = series[max_lag - lag : n - lag]

    return result.tolist()
        
        