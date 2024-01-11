"""Contains tests for the preprocessing module."""

import torch
from makemore_bigram.preprocessing import (
    create_one_hot_data,
    get_padded_data,
    pad_data,
    read_data,
)
from makemore_bigram.utils.paths import get_data_path

from makemore_bigram import INDEX_TO_TOKEN, N_TOKENS


def test_read_data() -> None:
    """Test the read_data function."""
    data_path = get_data_path()
    data = read_data(data_path)
    assert len(data) == 32033


def test_pad_data() -> None:
    """Test the pad_data and get_padded_data function."""
    data_path = get_data_path()
    data = read_data(data_path)
    padded_data = pad_data(data)
    assert len(padded_data) == 32033

    other_padded_data = get_padded_data()
    assert len(other_padded_data) == 32033

    assert padded_data == other_padded_data


def test_create_one_hot_data() -> None:
    """Test the pad_data and create_one_hot_data function."""
    data_path = get_data_path()

    encoded_input = create_one_hot_data(data_path=data_path)
    assert encoded_input.shape == torch.Size([260179, N_TOKENS])

    # Assert that the first name is .emma.
    assert INDEX_TO_TOKEN[encoded_input[0, :].argmax().item()] == "."
    assert INDEX_TO_TOKEN[encoded_input[1, :].argmax().item()] == "e"
    assert INDEX_TO_TOKEN[encoded_input[2, :].argmax().item()] == "m"
    assert INDEX_TO_TOKEN[encoded_input[3, :].argmax().item()] == "m"
    assert INDEX_TO_TOKEN[encoded_input[4, :].argmax().item()] == "a"
    assert INDEX_TO_TOKEN[encoded_input[5, :].argmax().item()] == "."
