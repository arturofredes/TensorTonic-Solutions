import numpy as np

def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    p_real = sigmoid(real_data @ D_W)
    p_fake = sigmoid(fake_data @ D_W)

    L_D = - (np.log(p_real) + np.log(1 - p_fake)).sum(axis = 1).mean()

    N = real_data.shape[0]
    residual = 1/N * (real_data.T @ (p_real - 1) + fake_data.T @ p_fake)

    D_W = D_W - learning_rate*residual
    return {"new_discriminator_weights":D_W,
            "discriminator_loss": L_D}