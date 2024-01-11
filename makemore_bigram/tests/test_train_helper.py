"""Contains tests for the train_helper module."""

import torch
from makemore_bigram.preprocessing import get_padded_data
from makemore_bigram.utils.train_helper import (
    create_bigram_count,
    create_count_matrix,
    get_count_matrix,
)

from makemore_bigram import N_TOKENS


def test_create_bigram_count() -> None:
    """Test that the create_bigram_count function is working."""
    padded_data = get_padded_data()
    bigram_count = create_bigram_count(padded_data=padded_data)
    assert len(bigram_count.keys()) == 627
    assert sum(bigram_count.values()) == 228146


def test_create_count_matrix() -> None:
    """Test the create_count_matrix and the get_count_matrix functions."""
    padded_data = get_padded_data()
    bigram_count = create_bigram_count(padded_data=padded_data)
    count_matrix = create_count_matrix(bigram_dict=bigram_count)
    assert count_matrix.shape == torch.Size([N_TOKENS, N_TOKENS])

    other_count_matrix = get_count_matrix()
    assert other_count_matrix.shape == torch.Size([N_TOKENS, N_TOKENS])

    assert (count_matrix == other_count_matrix).all().item()
