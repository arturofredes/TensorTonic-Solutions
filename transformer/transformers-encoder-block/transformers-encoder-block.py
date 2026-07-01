import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    m = np.mean(x, axis = -1, keepdims=True)
    s = np.std(x, axis = -1, keepdims=True)

    return gamma * (x - m) / np.sqrt(s ** 2 + eps) + beta
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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

    
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    h = x @ W1 + b1
    o = np.maximum(0, h)
    return o @ W2 + b2
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    out1 = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads) 
    x_prime = layer_norm(x + out1, gamma1, beta1)
    f = feed_forward(x_prime, W1, b1, W2, b2)
    out = layer_norm(x_prime + f, gamma2, beta2)
    return out