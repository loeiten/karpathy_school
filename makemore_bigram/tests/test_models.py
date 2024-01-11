"""Contains tests for the models module."""

import torch
from makemore_bigram.models import get_matrix_model, get_simple_neural_net

from makemore_bigram import N_TOKENS


def test_get_matrix_model() -> None:
    """Test the get_matrix_model function."""
    probability_matrix = get_matrix_model()
    assert probability_matrix.shape == torch.Size([N_TOKENS, N_TOKENS])


def test_get_simple_neural_net() -> None:
    """Test the get_simple_neural_net function."""
    matrix = get_simple_neural_net()
    assert matrix.shape == torch.Size([N_TOKENS, N_TOKENS])
