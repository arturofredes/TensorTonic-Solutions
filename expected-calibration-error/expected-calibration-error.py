import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    if n_bins<2 or n_bins>10000:
        return None
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_true = len(y_true)
    n_pred = len(y_pred)

    if n_true != n_pred:
        return None
    if n_true < 1 or n_true > 10000:
        return None

    if not all(yi in [0, 1] for yi in y_true):
        return None

    if not all((yi >= 0 and yi <= 1) for yi in y_pred):
        return None

    step = 1.0 / n_bins
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask_bin = [(y >= i * step and y <= (i + 1) * step) for y in y_pred]
        else:
            mask_bin = [(y >= i * step and y < (i + 1) * step) for y in y_pred]
            
        bin_pred = y_pred[mask_bin]
        bin_true = y_true[mask_bin]
        
        if len(bin_pred) == 0:
            continue
        
        acc = np.mean(bin_true)

        ece = ece + len(bin_pred) * np.abs(acc - np.mean(bin_pred))

    return ece / n_true
        
        

    