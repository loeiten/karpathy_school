"""Contains tests for the models module."""

import torch
from makemore_bigram.models import get_simple_neural_net

from makemore_bigram import N_TOKENS


def test_get_simple_neural_net() -> None:
    """Test the get_simple_neural_net function."""
    matrix = get_simple_neural_net()
    assert matrix.shape == torch.Size([N_TOKENS, N_TOKENS])
