"""Module for the tanh layer."""

from typing import Optional, Tuple

import torch
from makemore_wavenet.module import Module


class Tanh(Module):
    """Class mimicking the torch.nn.Tanh Module in PyTorch."""

    # Reducing the number of arguments here would be counter productive
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        """Initialize the output for the tanh layer.

        Args:
            device (Optional[torch.device], optional): Device to use for the tensors.
        """
        # NOTE: This is not implemented in the original torch layer, but is added
        #       for plotting convenience
        self.out = torch.empty(0, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> Tuple[torch.Tensor, ...]:
        """Return the parameters.

        Returns:
            Tuple[torch.Tensor,...]: The parameters of the layer
        """
        return tuple([])
