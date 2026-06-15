import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    L = max(len(seq) for seq in seqs)
    if L == 0:
        return np.array((0,0))
    if max_len is None: 
        max_len = L
    padded = np.full((len(seqs), max_len), pad_value)
    for i, seq in enumerate(seqs):
        l = len(seq)
        j = min(l, max_len)
        padded[i, :j] = seq[:j]

    return padded
        