import numpy as np

def moving_median(values: list, window_size: int) -> list:
    values = np.asarray(values)

    result = []

    for i in range(len(values) - window_size + 1):
        window = np.sort(values[i:i + window_size])

        if window_size % 2 == 1:
            median = window[window_size // 2]
        else:
            median = (
                window[window_size // 2 - 1] +
                window[window_size // 2]
            ) / 2

        result.append(median)

    return result