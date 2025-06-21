"""Module for the embedding layer."""

from typing import Optional, Tuple

import torch
from makemore_wavenet.module import Module


class Embedding(Module):
    """Class mimicking the torch.nn.Embedding Module in PyTorch."""

    # Reducing the number of arguments here would be counter productive
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        seed: int = 2147483647,
    ):
        """Set the weights for the embedding layer.

        Args:
            num_embeddings (int): Size of the dictionary of the embeddings.
            embedding_dim (int): The size of each embedding vector.
            device (Optional[torch.device], optional): Device to use for the tensors.
                Defaults to None.
            seed (int): The seed to use in the random number generator
        """
        g = torch.Generator(device=device).manual_seed(seed)
        self.weight = torch.randn(
            (num_embeddings, embedding_dim),
            generator=g,
            requires_grad=True,
            device=device,
        )
        # The size of out is determined during runtime
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        # NOTE: weights has dimension (num_embeddings, embedding_dim)
        #       x will in this case have the dimension (batch_size, block_size)
        #       weights[x] will grab embedding_dim vectors for each of the
        #       block_size characters
        #       The dimension of emb is therefore
        #       (batch_size, block_size, embedding_dim)
        self.out = self.weight[x]
        return self.out

    def parameters(self) -> Tuple[torch.Tensor, ...]:
        """Return the parameters.

        Returns:
            Tuple[torch.Tensor,...]: The parameters of the layer
        """
        params = [self.weight]
        return tuple(params)
