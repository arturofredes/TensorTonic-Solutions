import numpy as np
def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    w = 1.0
    comp_return = np.zeros((len(returns)))
    for i, r in enumerate(returns):
        w = w * (1 + r)
        comp_return[i] = w - 1

    return comp_return.tolist()