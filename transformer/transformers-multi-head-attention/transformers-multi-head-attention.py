import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch, seq , d_model = K.shape
    d_k = int(d_model / num_heads)

    QW = (Q @ W_q).reshape(batch, seq, num_heads, d_k)
    KW = (K @ W_k).reshape(batch, seq, num_heads, d_k)
    VW = (V @ W_v).reshape(batch, seq, num_heads, d_k)
    
    QW = QW.transpose(0, 2, 1, 3)  # (batch, num_heads, seq, d_k)
    KW = KW.transpose(0, 2, 1, 3)
    VW = VW.transpose(0, 2, 1, 3)
    
    S = (QW @ KW.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    A = softmax(S) @ VW

    MH = A.transpose(0, 2, 1, 3).reshape(batch, seq, d_model) @ W_o

    return MH

    
    