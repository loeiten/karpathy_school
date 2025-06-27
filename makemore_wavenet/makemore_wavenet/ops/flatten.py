"""Module for the FlattenConsecutive layer."""

from typing import Tuple

import torch
from makemore_wavenet.module import Module


class FlattenConsecutive(Module):
    """
    Class based the torch.nn.Flatten Module in PyTorch.

    NOTE: This will help us to create the dilated causal convolutional
        layer as consecutive flattening through the network will ensure
        that groups are gradually squeezed.
        This hinders us from squeezing all the information to the
        embedding at once, but will instead do so gradually

        This becomes the same as torch.nn.Flatten if
        n_consecutive_elements == block_size

    Example: We have
    >>> batch_size = 2  # How many names we consider at once
    >>> block_size = 3  # How many characters we consider at once
    >>> embedding_size = 3 # Dimension the characters has been squeezed into
    >>> input_data = torch.Tensor([[[1,4,7],
    ...                             [2,5,8],
    ...                             [3,6,9],
    ...                             [4,7,10]],
    ...                           [[11,15,19],
    ...                            [12,16,20],
    ...                            [13,17,21],
    ...                            [14,18,22]]])
    >>> # The first two characters from the first batch along the first
    >>> # dimension of the embedding
    >>> input_data[0,:,0]
    >>> # The first two characters from the first batch along the second
    >>> # dimension of the embedding
    >>> input_data[0,:,1]
    >>> n_consecutive_elements = 2  # Numbers we need to sum together
    >>> flattened = input_data.view(batch_size,
    ...                             block_size//n_consecutive_elements,
    ...                             embedding_size*n_consecutive_elements)
    >>> flattened[0,:,0]  # These numbers will be used first in the hierarchy
    >>> flattened[0,:,1]  # These  will be used next in the same level of the
    >>> # hierarchy
    """

    def __init__(self, n_consecutive_elements: int):
        """
        Set the number of consecutive elements.

        Args:
            n_consecutive_elements (int): Number of consecutive elements to
                group.
        """
        # The size of out is determined during runtime
        self.out = None
        self.n_consecutive_elements = n_consecutive_elements

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        batch_size, block_size, embedding_size = x.shape
        # Create the view which groups characters together
        x = x.view(
            batch_size,
            block_size // self.n_consecutive_elements,
            embedding_size * self.n_consecutive_elements,
        )
        if x.shape[1] == 1:
            # In the case where the middle dimension is 1, we remove it
            x = x.squeeze(1)
        self.out = x
        # Concatenate the vectors
        # x.shape[0] is the batch size
        # -1 means "infer the rest" (in this case block_size * n_embedding)
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self) -> Tuple[torch.Tensor, ...]:
        """Return the parameters.

        Returns:
            Tuple[torch.Tensor,...]: The parameters of the layer
        """
        return tuple([])
