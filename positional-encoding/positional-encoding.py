import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:, None]         
    dims = np.arange((d_model + 1) // 2)[None, :]         

    freqs = pos / (base ** (2 * dims / d_model))

    pe = np.empty((seq_len, d_model))
    pe[:, 0::2] = np.sin(freqs)
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(freqs)
    else:
        pe[:, 1::2] = np.cos(freqs[:,:-1])
    return pe