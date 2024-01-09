"""Module for loading and preprocessing data."""

from pathlib import Path
from typing import Dict, Tuple

import torch
from makemore_bigram.utils.paths import get_data_path

from makemore_bigram import ALPHABET_DICT


def read_data(data_path: Path) -> Tuple:
    """Return the data as a tuple.

    Args:
        data_path (Path): Path to the data

    Returns:
        Tuple: The data as a list
    """
    return tuple(data_path.open("r").read().splitlines())


def pad_data(data_tuple: Tuple) -> Tuple:
    """Pad the data with start and stop tokens.

    We could chose to have a separate start and stop token.
    However, having simply one token suffices.
    We will use '.' as this token.

    Args:
        data_tuple (Tuple): The unprocessed data

    Returns:
        Tuple: The padded data
    """
    padded_data = [f".{name}." for name in data_tuple]
    return tuple(padded_data)


def create_bigram_count(padded_data: Tuple) -> Dict[Tuple[str, str], int]:
    """Create a dict where the keys are bigrams and the value is the count.

    Args:
        padded_data (Tuple): The data padded with start and stop tokens

    Returns:
        Dict[Tuple[str, str], int]: Bigram count of the data
    """
    bigram_dict: Dict[Tuple[str, str], int] = {}
    for padded_name in padded_data:
        for first_token, second_token in zip(padded_name, padded_name[1:]):
            bigram = (first_token, second_token)
            # bigram_dict.get(bigram, 0) gets the count, if bigram is not present in
            # the dict it will return 0
            bigram_dict[bigram] = bigram_dict.get(bigram, 0) + 1
    return bigram_dict


def create_count_matrix(bigram_dict: Dict[Tuple[str, str], int]) -> torch.Tensor:
    """Create the count matrix.

    The rows of the count matrix are the first token of the bigram and the
    columns are the following token.
    The elements represents the frequency in the dataset

    Args:
        bigram_dict (Dict[Tuple[str, str], int]): Bigram count of the data

    Returns:
        torch.Tensor: The count matrix
    """
    count_matrix = torch.zeros((28, 28), dtype=torch.int32)

    for bigram, count in bigram_dict.items():
        first_token = bigram[0]
        second_token = bigram[1]
        row_idx = ALPHABET_DICT[first_token]
        col_idx = ALPHABET_DICT[second_token]
        count_matrix[row_idx, col_idx] = count
    return count_matrix


def get_count_matrix() -> torch.Tensor:
    """
    Return the count tensor for the data.

    Returns:
        torch.Tensor: The count matrix
    """
    data_path = get_data_path()
    data_tuple = read_data(data_path=data_path)
    padded_data = pad_data(data_tuple=data_tuple)
    bigram_count = create_bigram_count(padded_data=padded_data)
    count_matrix = create_count_matrix(bigram_dict=bigram_count)
    return count_matrix
