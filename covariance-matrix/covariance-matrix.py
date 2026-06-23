import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    
    if len(X.shape)!=2:
        return None

    N = X.shape[0]
    if N < 2:
        return None

    mu = np.mean(X, axis=0, keepdims=True)
    X_c = X - mu

    return (1/(N - 1)) * X_c.T @ X