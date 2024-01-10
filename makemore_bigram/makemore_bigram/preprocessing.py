"""Module for loading and preprocessing data."""

from pathlib import Path
from typing import Dict, Tuple

import torch
from makemore_bigram.utils.paths import get_data_path

from makemore_bigram import ALPHABET_DICT


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


def create_bigram_count(padded_data: Tuple[str, ...]) -> Dict[Tuple[str, str], int]:
    """Create a dict where the keys are bigrams and the value is the count.

    Args:
        padded_data (Tuple[str, ...]): The data padded with start and stop tokens

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


def create_probability_matrix(count_matrix: torch.Tensor) -> torch.Tensor:
    """Create a probability matrix from the count matrix.

    Args:
        count_matrix (torch.Tensor): The matrix containing the frequency of the bigrams

    Returns:
        torch.Tensor: The probability matrix
    """
    # We add one to the probability matrix to avoid dividing by zero
    probability_matrix = (count_matrix + 1).float()

    # We take the sum along the rows (axis=1)
    # We have keep_dims=True due to the broadcasting rules
    # The count matrix has dim 27, 27
    # .sum will create a matrix with dims 27, 1
    # Broadcasting rules will copy .sum to be 27 ->, 27
    # If we had keep_dims=False we would get dims 0, 27
    # Broadcasting rules would the copy .sum to be 27, <- 27
    probability_matrix /= probability_matrix.sum(dim=1, keepdim=True)
    return probability_matrix


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


def get_count_matrix() -> torch.Tensor:
    """
    Return the count tensor for the data.

    Returns:
        torch.Tensor: The count matrix
    """
    padded_data = get_padded_data()
    bigram_count = create_bigram_count(padded_data=padded_data)
    count_matrix = create_count_matrix(bigram_dict=bigram_count)
    return count_matrix


def get_probability_matrix() -> torch.Tensor:
    """
    Return the probability tensor for the data.

    Returns:
        torch.Tensor: The probability matrix
    """
    count_matrix = get_count_matrix()
    probability_matrix = create_probability_matrix(count_matrix=count_matrix)
    return probability_matrix
