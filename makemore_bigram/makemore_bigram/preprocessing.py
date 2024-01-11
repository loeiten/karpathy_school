"""Module for loading and preprocessing data."""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from makemore_bigram.utils.paths import get_data_path


def read_data(data_path: Path) -> Tuple[str, ...]:
    """Return the data as a tuple.

    Args:
        data_path (Path): Path to the data

    Returns:
        Tuple[str, ...]: The data as a list
    """
    return tuple(data_path.open("r").read().splitlines())


def pad_data(data_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
    """Pad the data with start and stop tokens.

    We could chose to have a separate start and stop token.
    This would lead to the matrix having a row with only zeros (starting with an
    end token), and a column with just zeros (the start token following a
    character).
    We can also notice that only one token can do the job for both start and
    stop.
    We have used '.' as this token in ALPHABET_DICT.

    Args:
        data_tuple (Tuple[str, ...]): The unprocessed data

    Returns:
        Tuple: The padded data
    """
    padded_data = [f".{name}." for name in data_tuple]
    return tuple(padded_data)


def get_padded_data() -> Tuple[str, ...]:
    """
    Return the padded data.

    Returns:
        Tuple[str,...]: The names padded
    """
    data_path = get_data_path()
    data_tuple = read_data(data_path=data_path)
    padded_data = pad_data(data_tuple=data_tuple)
    return padded_data


def get_bigram_ground_truth() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the ground truth for the bigrams.

    Returns:
        torch.Tensor: The input character (one hot encoded tokens)
        torch.Tensor: The output character (one hot encoded tokens)
    """
    padded_data = get_padded_data()
    input_token: List[str] = []
    output_token: List[str] = []

    for name in padded_data:
        for token_1, token_2 in zip(name, name[1:]):
            input_token.append(token_1)
            output_token.append(token_2)

    # NOTE: torch.tensor is a function, whilst torch.Tensor is a constructor
    #       which infers the dtype
    input_tensor = torch.tensor(input_token)
    output_tensor = torch.tensor(output_token)

    # One hot encoding of the data
    # pylint: disable-next=not-callable
    encoded_input = F.one_hot(input_tensor, num_classes=27)
    # pylint: disable-next=not-callable
    encoded_output = F.one_hot(output_tensor, num_classes=27)

    return encoded_input, encoded_output
