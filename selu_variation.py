import numpy as np

def selu_variation(x, alpha=1.67326, lambda_=1.0507, beta=1.0, gamma=0.1, omega=2.0):
    """
    SELU Variation Activation Function implemented with NumPy.
    
    Parameters:
    - x: Input data.
    - alpha: Scaling parameter for exponential growth for negative inputs.
    - lambda_: Scaling parameter to ensure self-normalization.
    - beta: Adjustment for the exponential component.
    - gamma: Amplitude of the sinusoidal component.
    - omega: Frequency of the sinusoidal component.
    
    Returns:
    - The activated output following the SELU variation formula.
    """
    selu_var = np.where(x > 0, lambda_ * x,
                        lambda_ * (alpha * (np.exp(beta * x) - 1) + gamma * np.sin(omega * x)))
    return selu_var
