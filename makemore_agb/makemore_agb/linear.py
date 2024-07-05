"""Module for the linear layer."""

from typing import List, Optional

import torch
from makemore_agb.module import Module


class Linear(Module):
    """Class mimicking the torch.nn.Linear Module in PyTorch."""

    # Reducing the number of arguments here would be counter productive
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        seed: int = 2147483647,
    ):
        """Set the weights and biases for the linear layer.

        Args:
            fan_in (int): Number of inputs to the layer.
            fan_out (int): Number of outputs from the layer.
            bias (bool, optional): Whether or not to use the bias term.
                Defaults to True.
            device (Optional[torch.device], optional): Device to use for the tensors.
                Defaults to None.
            seed (int): The seed to use in the random number generator
        """
        g = torch.Generator(device=device).manual_seed(seed)
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        # NOTE: This is not implemented in the original torch layer, but is added
        #       for plotting convenience
        self.out = torch.empty(fan_out, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return the parameters.

        Returns:
            List[torch.Tensor]: The parameters of the layer
        """
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
