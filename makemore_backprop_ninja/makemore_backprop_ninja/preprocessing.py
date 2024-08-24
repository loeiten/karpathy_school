"""Module for loading and preprocessing data."""

from pathlib import Path
from typing import List, Tuple

import torch
from makemore_backprop_ninja.utils.paths import get_data_path

from makemore_backprop_ninja import DATASET, DEVICE, TOKEN_TO_INDEX


def read_data(data_path: Path) -> Tuple[str, ...]:
    """Return the data as a tuple.

    Args:
        data_path (Path): Path to the data

    Returns:
        Tuple[str, ...]: The data as a list
    """
    return tuple(data_path.open("r").read().splitlines())


def pad_data(data_tuple: Tuple[str, ...], block_size: int) -> Tuple[str, ...]:
    """Pad the data with start and stop tokens.

    We could chose to have a separate start and stop token.
    We can also notice that only one token can do the job for both start and
    stop.
    We have used '.' as this token in ALPHABET_DICT.

    Args:
        data_tuple (Tuple[str, ...]): The unprocessed data
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length

    Returns:
        Tuple: The padded data
    """
    padded_data = [f"{'.'*block_size}{name}." for name in data_tuple]
    return tuple(padded_data)


def create_feature_and_labels(
    input_data: Tuple[str, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the training data.

    NOTE: We do not need to one-hot encode
          If we have a embedding matrix
          C = torch.rand(vocab_size, embedding_size)
          Then
          C[token_idx]
          is equivalent to
          F.one_hot(torch.tensor(token_idx), num_classes=vocab_size).float @ C

    NOTE: The embedding look-up is fairly flexible in PyTorch as we can index
          the embedding matrix with C[torch.tensor]

    Args:
        input_data (Tuple[str,...]): The data to create the features and labels
            from

    Returns:
        torch.Tensor: The input characters
        torch.Tensor: The ground truth prediction
    """
    # Infer the block_size
    # Subtract one for the stop token
    block_size = input_data[0].count(".") - 1

    input_tokens_list: List[List[int]] = []
    output_token_list: List[int] = []

    for name in input_data:
        # Fill the initial context
        context: List[int] = []
        for token in name[:block_size]:
            context.append(TOKEN_TO_INDEX[token])
        # Continue with the rest of the name
        for token in name[block_size:]:
            # Store the current context as the input
            input_tokens_list.append(context)

            token_as_index = TOKEN_TO_INDEX[token]
            # Store the next token as the label
            output_token_list.append(token_as_index)

            # Crop the context, and add the new token
            context = context[1:] + [token_as_index]

    # NOTE: torch.tensor is a function, whilst torch.Tensor is a constructor
    #       which infers the dtype
    input_tensor = torch.tensor(input_tokens_list, device=DEVICE)
    output_tensor = torch.tensor(output_token_list, device=DEVICE)

    return input_tensor, output_tensor


def get_padded_data(block_size: int) -> Tuple[str, ...]:
    """
    Return the padded data.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length

    Returns:
        Tuple[str,...]: The names padded
    """
    data_path = get_data_path()
    data_tuple = read_data(data_path=data_path)
    padded_data = pad_data(data_tuple=data_tuple, block_size=block_size)
    return padded_data


def get_dataset(
    block_size: int = 3,
    seed: int = 2147483647,
) -> DATASET:
    """Get the train, validation and test set data.

    Args:
        block_size (int, optional): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length. Defaults to 3.
        seed (int): The seed to use

    Returns:
        The dataset.
        The input data is the data fed into the model
        The ground truth are the expected data from the model
    """
    torch.manual_seed(seed)

    padded_data = get_padded_data(block_size=block_size)
    input_data, output_data = create_feature_and_labels(input_data=padded_data)

    # Get the permutation to use as indices
    # See
    # https://stackoverflow.com/a/73187955/2786884
    # For details
    # Accessing the only element in the shape
    indices = torch.randperm(output_data.shape[0], device=DEVICE)
    # Shuffle the data
    shuffled_input_data = input_data[indices]
    shuffled_output_data = output_data[indices]

    # Split the data in 80-10-10
    n1 = int(0.8 * shuffled_output_data.shape[0])
    n2 = int(0.9 * shuffled_output_data.shape[0])

    dataset: DATASET = {}

    dataset["training_input_data"] = shuffled_input_data[:n1]
    dataset["training_ground_truth"] = shuffled_output_data[:n1]

    dataset["validation_input_data"] = shuffled_input_data[n1:n2]
    dataset["validation_ground_truth"] = shuffled_output_data[n1:n2]

    dataset["test_input_data"] = shuffled_input_data[n2:]
    dataset["test_ground_truth"] = shuffled_output_data[n2:]

    return dataset
