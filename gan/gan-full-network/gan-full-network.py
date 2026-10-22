import numpy as np

def gan_forward(
    z: np.ndarray,
    real_data: np.ndarray,
    G_W: np.ndarray,
    G_b: np.ndarray,
    D_W: np.ndarray,
) -> dict:
    """
    Returns generated samples, probabilities, and both GAN losses.
    """
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))
    # generate samples
    x = np.tanh(z @ G_W + G_b)

    #generate probabilities
    real_probs = sigmoid(real_data @ D_W)
    fake_probs = sigmoid(x @ D_W)

    eps = 1e-8

    real_probs = np.clip(real_probs, eps, 1 - eps)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    # Discriminator loss
    l = np.log(real_probs) + np.log(1 - fake_probs)
    LD = -np.mean(l)

    # Generator loss
    l = np.log(fake_probs)
    LG = -np.mean(l)

    return {
        "generated_samples": x,
        "real_probabilities": real_probs,
        "fake_probabilities": fake_probs,
        "discriminator_loss": float(LD),
        "generator_loss": float(LG),
    }