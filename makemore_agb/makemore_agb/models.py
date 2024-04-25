"""Module for models."""

from typing import Tuple

import torch

from makemore_agb import VOCAB_SIZE


def get_model(
    block_size: int,
    embedding_size: int = 2,
    hidden_layer_neurons: int = 100,
    seed: int = 2147483647,
) -> Tuple[torch.Tensor, ...]:
    """Return the model.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length
        embedding_size (int): The size of the embedding
        hidden_layer_neurons (int): The seed for the random number generator
        seed (int): The seed for the random number generator

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the parameters of the
            neural net.
    """
    g = torch.Generator().manual_seed(seed)

    # NOTE: randn draws from normal distribution, whereas rand draws from a
    #       uniform distribution
    c = torch.randn((VOCAB_SIZE, embedding_size), generator=g, requires_grad=True)
    w1 = torch.randn(
        (block_size * embedding_size, hidden_layer_neurons),
        generator=g,
        requires_grad=True,
    )
    b1 = torch.randn(hidden_layer_neurons, generator=g, requires_grad=True)
    w2 = torch.randn(
        (hidden_layer_neurons, VOCAB_SIZE), generator=g, requires_grad=True
    )
    b2 = torch.randn(VOCAB_SIZE, generator=g, requires_grad=True)
    parameters = (c, w1, b1, w2, b2)

    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return parameters
