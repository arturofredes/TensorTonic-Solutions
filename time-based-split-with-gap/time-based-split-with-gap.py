import numpy as np

def time_based_split(timestamps: list, train_end: int, gap: int, val_end: int) -> dict:
    """
    Returns a dictionary with train_idx and val_idx as sorted integer NumPy arrays.
    """
    timestamps = np.asarray(timestamps)

    train_mask = timestamps < train_end
    val_mask = (timestamps >= train_end + gap) & (timestamps < val_end)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx
    }