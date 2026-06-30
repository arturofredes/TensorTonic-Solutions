import numpy as np

def color_to_grayscale(image):
    image = np.asarray(image)

    # Already grayscale
    if image.ndim == 2:
        return image.tolist()

    # Single-channel image (H, W, 1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0].tolist()

    # RGB image (H, W, 3)
    if image.ndim == 3 and image.shape[-1] == 3:
        gray = (
            image[..., 0] * 0.299 +
            image[..., 1] * 0.587 +
            image[..., 2] * 0.114
        )
        return gray.tolist()

    raise ValueError("Unsupported image format")