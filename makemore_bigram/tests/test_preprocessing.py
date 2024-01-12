"""Contains tests for the preprocessing module."""

import torch
from makemore_bigram.preprocessing import (
    create_training_data,
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


def test_create_training_data() -> None:
    """Test the pad_data and create_one_hot_data function."""
    padded_data = get_padded_data()

    encoded_input, ground_truth = create_training_data(input_data=padded_data)
    assert encoded_input.shape == torch.Size([228146, N_TOKENS])
    assert ground_truth.shape == torch.Size([228146])

    # Assert that the first name is .emma.
    assert INDEX_TO_TOKEN[encoded_input[0, :].argmax().item()] == "."
    assert INDEX_TO_TOKEN[encoded_input[1, :].argmax().item()] == "e"
    assert INDEX_TO_TOKEN[encoded_input[2, :].argmax().item()] == "m"
    assert INDEX_TO_TOKEN[encoded_input[3, :].argmax().item()] == "m"
    assert INDEX_TO_TOKEN[encoded_input[4, :].argmax().item()] == "a"
    assert INDEX_TO_TOKEN[encoded_input[5, :].argmax().item()] == "."

    assert INDEX_TO_TOKEN[ground_truth[0].item()] == "e"
    assert INDEX_TO_TOKEN[ground_truth[1].item()] == "m"
    assert INDEX_TO_TOKEN[ground_truth[2].item()] == "m"
    assert INDEX_TO_TOKEN[ground_truth[3].item()] == "a"
    assert INDEX_TO_TOKEN[ground_truth[4].item()] == "."
