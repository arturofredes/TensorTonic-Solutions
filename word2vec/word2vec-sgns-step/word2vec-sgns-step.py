import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # Get embeddings
    v_c = W_in[center_id].clone()      # (d,)
    u_o = W_out[pos_id].clone()        # (d,)
    u_n = W_out[neg_ids].clone()       # (k, d)

    # Scores
    s_o = torch.dot(v_c, u_o)          # scalar
    s_n = u_n @ v_c                    # (k,)

    # Sigmoids
    sig_o = torch.sigmoid(s_o)         # scalar
    sig_n = torch.sigmoid(s_n)         # (k,)

    # Gradients w.r.t. scores act as scalar coefficients
    grad_coeff_o = (sig_o - 1)         # scalar, for positive sample
    grad_coeff_n = sig_n               # (k,), for negative samples

    # ∇u_o = (σ(s_o) - 1) * v_c
    grad_u_o = grad_coeff_o * v_c                     # (d,)

    # ∇u_ni = σ(s_ni) * v_c   for each negative i
    grad_u_n = grad_coeff_n.unsqueeze(1) * v_c.unsqueeze(0)  # (k, d)

    # ∇v_c = (σ(s_o)-1)*u_o + Σ_i σ(s_ni)*u_ni
    grad_v_c = grad_coeff_o * u_o + grad_coeff_n @ u_n       # (d,)

    # SGD updates
    W_in[center_id] -= lr * grad_v_c
    W_out[pos_id]   -= lr * grad_u_o
    W_out.index_add_(0, neg_ids, -lr * grad_u_n)

    return W_in, W_out
    
