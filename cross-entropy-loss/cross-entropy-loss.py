import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = y_true.shape[0]
    if N != y_pred.shape[0]:
        raise ValueError("dimensions do not match")

    # Pick out y_pred[i, y_true[i]] for every i at once
    correct_class_probs = y_pred[np.arange(N), y_true]
    return -np.mean(np.log(correct_class_probs))
        