import numpy as np
def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    # Write code here
    n = len(values)
    aux = np.zeros((n - window_size + 1, window_size))
    for i in range(n - window_size + 1):
        aux[i,:] = values[i: i + window_size]

    return aux.mean(axis = 1).tolist()
    
    
    