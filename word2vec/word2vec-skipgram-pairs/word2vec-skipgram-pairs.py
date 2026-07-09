import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    n = token_ids.shape[0]
    pairs = []
    for i, token in enumerate(token_ids):
        left = max(0, i - window)
        right = min(n - 1, i + window)
        for j in range(left, right + 1):
            if j == i:
                continue
            pairs.append([token,token_ids[j]])

    return torch.tensor(pairs, dtype = torch.int64).reshape(-1, 2)
            