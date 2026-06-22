import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """

    param = np.asarray(param)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)
    
    # update first moment
    m_new = beta1 * m + (1 - beta1) * grad

    # update second moment
    v_new = beta2* v + (1 - beta2) * grad ** 2

    #bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Parameter Update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return (param_new, m_new, v_new)
