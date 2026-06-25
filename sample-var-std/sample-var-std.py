import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    m = np.mean(x)

    s2 = (1/(x.shape[0] - 1)) * (x - m).T @ (x - m)

    return (s2, np.sqrt(s2))