"""Module containing helping function for training."""

from typing import Dict, Tuple

import torch
from makemore_bigram.preprocessing import get_padded_data

from makemore_bigram import TOKEN_TO_INDEX


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
        row_idx = TOKEN_TO_INDEX[first_token]
        col_idx = TOKEN_TO_INDEX[second_token]
        count_matrix[row_idx, col_idx] = count
    return count_matrix


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
