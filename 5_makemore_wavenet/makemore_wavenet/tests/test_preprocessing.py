"""Contains tests for the preprocessing module."""

import torch
from makemore_wavenet.preprocessing import (
    create_feature_and_labels,
    get_dataset,
    get_padded_data,
    pad_data,
    read_data,
)
from makemore_wavenet.utils.paths import get_data_path

from makemore_wavenet import INDEX_TO_TOKEN


def test_read_data() -> None:
    """Test the read_data function."""
    data_path = get_data_path()
    data = read_data(data_path=data_path)
    assert len(data) == 32033


def test_pad_data() -> None:
    """Test the pad_data and get_padded_data function."""
    data_path = get_data_path()
    data = read_data(data_path=data_path)
    padded_data = pad_data(data_tuple=data, block_size=3)
    assert len(padded_data) == 32033

    other_padded_data = get_padded_data(block_size=3)
    assert len(other_padded_data) == 32033

    assert padded_data == other_padded_data


def test_create_feature_and_labels() -> None:
    """Test the pad_data and create_feature_and_labels function."""
    block_size = 3
    padded_data = get_padded_data(block_size=block_size)

    input_tensor, output_tensor = create_feature_and_labels(input_data=padded_data)
    assert input_tensor.shape == torch.Size([228146, block_size])
    assert output_tensor.shape == torch.Size([228146])

    # Assert that the first name is ...emma
    # ...
    assert INDEX_TO_TOKEN[input_tensor[0, 0].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[0, 1].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[0, 2].item()] == "."
    # ..e
    assert INDEX_TO_TOKEN[input_tensor[1, 0].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[1, 1].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[1, 2].item()] == "e"
    # .em
    assert INDEX_TO_TOKEN[input_tensor[2, 0].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[2, 1].item()] == "e"
    assert INDEX_TO_TOKEN[input_tensor[2, 2].item()] == "m"
    # emm
    assert INDEX_TO_TOKEN[input_tensor[3, 0].item()] == "e"
    assert INDEX_TO_TOKEN[input_tensor[3, 1].item()] == "m"
    assert INDEX_TO_TOKEN[input_tensor[3, 2].item()] == "m"
    # mma
    assert INDEX_TO_TOKEN[input_tensor[4, 0].item()] == "m"
    assert INDEX_TO_TOKEN[input_tensor[4, 1].item()] == "m"
    assert INDEX_TO_TOKEN[input_tensor[4, 2].item()] == "a"
    # Next name
    # ...
    assert INDEX_TO_TOKEN[input_tensor[5, 0].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[5, 1].item()] == "."
    assert INDEX_TO_TOKEN[input_tensor[5, 2].item()] == "."

    assert INDEX_TO_TOKEN[output_tensor[0].item()] == "e"
    assert INDEX_TO_TOKEN[output_tensor[1].item()] == "m"
    assert INDEX_TO_TOKEN[output_tensor[2].item()] == "m"
    assert INDEX_TO_TOKEN[output_tensor[3].item()] == "a"
    assert INDEX_TO_TOKEN[output_tensor[4].item()] == "."


def test_get_dataset():
    """Test the pad_data and create_feature_and_labels function."""
    block_size = 3

    dataset = get_dataset(block_size=block_size)

    # Assert the shapes
    assert dataset["training_input_data"].shape == torch.Size([182516, block_size])
    assert dataset["training_ground_truth"].shape == torch.Size([182516])
    assert dataset["validation_input_data"].shape == torch.Size([22815, block_size])
    assert dataset["validation_ground_truth"].shape == torch.Size([22815])
    assert dataset["test_input_data"].shape == torch.Size([22815, block_size])
    assert dataset["test_ground_truth"].shape == torch.Size([22815])
