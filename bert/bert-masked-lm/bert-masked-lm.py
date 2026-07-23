import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    labels = np.full_like(token_ids, -100)
    masked_ids = token_ids.copy()
    for i in range(token_ids.shape[0]):
        for j in range(token_ids[i].shape[0]):
            if mask_positions[i][j]:
                labels[i][j] = token_ids[i][j]
                if replace_probs[i][j] < 0.8:
                    masked_ids[i][j] = mask_token_id
                elif replace_probs[i][j] >= 0.9:
                    continue
                else:
                    masked_ids[i][j] = random_tokens[i][j]
            else:
                continue

    return masked_ids, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        # YOUR CODE HERE
        return hidden_states @ self.W + self.b 
