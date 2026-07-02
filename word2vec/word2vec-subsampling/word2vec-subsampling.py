import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    N = counts.sum()
    freqs = counts/N
    score = torch.sqrt( t / freqs)

    p = torch.clamp(score, max=1.0)
    return p
