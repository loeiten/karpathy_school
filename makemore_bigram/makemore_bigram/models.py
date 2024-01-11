"""Module for models."""

import torch
from makemore_bigram.train import get_probability_matrix

from makemore_bigram import N_TOKENS


def get_matrix_model() -> torch.Tensor:
    """Return the (trained) matrix model.

    Returns:
        torch.Tensor: The probability model.
    """
    return get_probability_matrix()


def get_simple_neural_net(seed: int = 2147483647) -> torch.Tensor:
    """Return the neural net model.

    This model only has one hidden layer with no biases.

    Args:
        seed (int): The seed for the random number generator

    Returns:
        torch.Tensor: The parameters of the neural net.
    """
    g = torch.Generator().manual_seed(seed)
    weights = torch.rand((N_TOKENS, N_TOKENS), generator=g)
    return weights
