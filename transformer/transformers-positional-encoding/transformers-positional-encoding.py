import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    base=10000.0
    pos = np.arange(seq_length)[:, None]         
    dims = np.arange((d_model + 1) // 2)[None, :]         

    freqs = pos / (base ** (2 * dims / d_model))

    pe = np.empty((seq_length, d_model))
    pe[:, 0::2] = np.sin(freqs)
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(freqs)
    else:
        pe[:, 1::2] = np.cos(freqs[:,:-1])
    return pe