"""Module for the tanh layer."""

from typing import Optional, Tuple

import torch
from makemore_wavenet.module import Module


# Reducing the number of attributes here will penalize the didactical purpose
class BatchNorm1d(Module):
    """Class mimicking the torch.nn.BatchNorm1d Module in PyTorch."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """Initialize the parameters and the buffers for the normalization.

        Args:
            dim (int): The size of the layer
            eps (float, optional): Epsilon to avoid division by zero. Defaults to 1e-5.
            momentum (float, optional): Momentum to use in the buffer update.
                Defaults to 0.1.
            device (Optional[torch.device], optional): Device to use for the tensors.
        """
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Initialize the parameters
        # The batch normalization gain
        self.gamma = torch.ones(dim, device=device)
        # The batch normalization bias
        self.beta = torch.zeros(dim, device=device)

        # Initialize the buffers
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

        # Initialize the output
        self.out = torch.empty(dim, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        # Forward pass
        if self.training:
            # Mean of the batch
            x_mean = x.mean(0, keepdim=True)
            # Variance of the batch
            x_var = x.var(0, keepdim=True)
        else:
            # Use the buffers
            x_mean = self.running_mean
            x_var = self.running_var

        # Normalize to unit variance
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        # Update the running buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * x_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self) -> Tuple[torch.Tensor, ...]:
        """Return the parameters.

        Returns:
            Tuple[torch.Tensor,...]: The parameters of the layer
        """
        return tuple([self.gamma, self.beta])
