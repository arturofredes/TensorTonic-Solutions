import numpy as np
def linear_interpolation(values: list) -> list:
    """
    Returns a copy with every missing value interpolated.
    """
    i = 0 
    n = len(values)
    while i < n:
        if values[i] is not None:
            i += 1 
            continue
        else:
            left = values[i - 1]
            i_l = i - 1
            right = None
            while right is None:
                right = values[i + 1]
                i += 1
                
            diff = right - left
            for j in range(i_l, i):
                values[j] = left + (j - i_l) / (i - i_l) * diff

    return values
        