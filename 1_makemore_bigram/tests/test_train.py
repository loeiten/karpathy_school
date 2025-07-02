"""Contains tests for the train module."""

import torch
from makemore_bigram.train import get_neural_net, get_probability_matrix

from makemore_bigram import N_TOKENS


def test_get_probability_matrix() -> None:
    """Test the get_probability_matrix function."""
    matrix = get_probability_matrix()
    assert matrix.shape == torch.Size([N_TOKENS, N_TOKENS])


def test_get_neural_net() -> None:
    """Test the test_get_neural_net function."""
    matrix = get_neural_net(epochs=1)
    assert matrix.shape == torch.Size([N_TOKENS, N_TOKENS])
