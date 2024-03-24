# SELU Variation Activation Function

The SELU Variation is designed to enhance the self-normalizing properties of the standard SELU function by introducing a non-monotonic behavior for negative inputs through a sinusoidal component.

<img width="563" alt="selu_variation" src="https://github.com/ToyMath/SELUVariation/assets/5700430/3290aa5e-6dd0-4bf1-a7bf-dcbed16cf406">

## Mathematical Formula

The SELU Variation activation function is defined as:

```math
f(x) = \lambda \cdot \left\{ \begin{array}{ll} x & \text{if } x > 0\\ \alpha \cdot (e^{\beta \cdot x} - 1) + \gamma \cdot \sin(\omega \cdot x) & \text{if } x \leq 0 \end{array} \right.
```

Where:
- \( $\\lambda \$) is a scaling parameter to ensure self-normalization.
- \( $\\alpha \$) and \( $\\beta\$ \) adjust the exponential growth for negative inputs.
- \( $\\gamma \$) and \( $\\omega\$ \) control the amplitude and frequency of the sinusoidal component.

### Code

```python
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
```

## Installation

```bash
git clone https://github.com/ToyMath/SELUVariation.git
cd SELUVariation
```

## Usage

### TensorFlow Implementation

```python
import tensorflow as tf

class SELUVariation(tf.keras.layers.Layer):
    def __init__(self, alpha=1.67326, lambda_=1.0507, beta=1.0, gamma=0.1, omega=2.0, **kwargs):
        super(SELUVariation, self).__init__(**kwargs)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    def call(self, inputs):
        return tf.where(inputs > 0, self.lambda_ * inputs,
                        self.lambda_ * (self.alpha * (tf.exp(self.beta * inputs) - 1) + self.gamma * tf.sin(self.omega * inputs)))
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class SELUVariation(nn.Module):
    def __init__(self, alpha=1.67326, lambda_=1.0507, beta=1.0, gamma=0.1, omega=2.0):
        super(SELUVariation, self).__init__()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    def forward(self, inputs):
        positive_part = self.lambda_ * inputs
        negative_part = self.lambda_ * (self.alpha * (torch.exp(self.beta * inputs) - 1) + self.gamma * torch.sin(self.omega * inputs))
        return torch.where(inputs > 0, positive_part, negative_part)
```

### JAX Implementation

```python
import jax.numpy as jnp
from jax import jit

@jit
def selu_variation(x, lambda_=1.0507, alpha=1.67326, beta=1.0, gamma=0.1, omega=2.0):
    """
    SELU Variation Activation Function implemented in JAX.

    Parameters:
    - x: The input tensor.
    - lambda_: Scaling parameter λ to ensure self-normalization.
    - alpha, beta: Adjust the exponential growth for negative inputs.
    - gamma, omega: Control the amplitude and frequency of the sinusoidal component.

    Returns:
    - The activated output following the SELU variation formula.
    """
    positive_part = lambda_ * x
    negative_part = lambda_ * (alpha * (jnp.exp(beta * x) - 1) + gamma * jnp.sin(omega * x))
    return jnp.where(x > 0, positive_part, negative_part)

```

## Customization

You can adjust the SELU Variation activation function parameters (`lambda`, `alpha`, `beta`, `gamma`, `omega`) to fine-tune its behavior for your specific tasks.

## Citation

If you use SELUVariation in your research, please cite the following work:

```bibtex
@misc{SELUVariation-2024,
  author = {Aakash Apoorv},
  title = {SELUVariation},
  year = {2024},
  howpublished = {\url{https://github.com/ToyMath/SELUVariation}},
}
```
