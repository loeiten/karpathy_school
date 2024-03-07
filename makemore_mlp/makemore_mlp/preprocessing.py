"""Module for loading and preprocessing data."""

from pathlib import Path
from typing import List, Tuple

import torch
from makemore_mlp.utils.paths import get_data_path

from makemore_mlp import TOKEN_TO_INDEX


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
    input_tensor = torch.tensor(input_tokens_list)
    output_tensor = torch.tensor(output_token_list)

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
