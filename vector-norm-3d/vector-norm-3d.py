import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.array(v)
    return np.sqrt(np.sum(v*v, axis = -1))