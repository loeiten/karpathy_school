"""Module for the flatten layer."""

from typing import List

import torch
from makemore_wavenet.module import Module


class Flatten(Module):
    """Class mimicking the torch.nn.Flatten Module in PyTorch."""

    def __init__(
        self,
    ):
        """Set the out for the flatten layer."""
        # The size of out is determined during runtime
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        # Concatenate the vectors
        # x.shape[0] is the batch size
        # -1 means "infer the rest" (in this case block_size * n_embedding)
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return the parameters.

        Returns:
            List[torch.Tensor]: The parameters of the layer
        """
        return []
