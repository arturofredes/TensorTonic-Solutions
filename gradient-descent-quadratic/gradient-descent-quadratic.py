def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    for _ in range(steps):
        x0 = x0 - lr * (2 * a * x0 + b)

    return float(x0)