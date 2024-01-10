"""Module containing metric calculation."""

from typing import Tuple

import torch

from makemore_bigram import ALPHABET_DICT


def calculate_avg_nll_of_matrix_model(
    data: Tuple[str, ...], model: torch.Tensor
) -> float:
    """Calculate the average negative log likelihood (nll) of the matrix model.

    In the model we would like to maximize the likelihood of the data with
    respect to the model parameters.
    As the likelihood is obtained by multiplying the probabilities we can obtain
    very small numbers which can lead to high truncation errors.
    Hence, we can instead maximize the log of the likelihood.
    This is equivalent to minimizing the negative log likelihood, which is the
    same as minimizing the average of the log likelihood.

    A model with average nll = 0 is a model which perfectly fits the data

    Args:
        data (Tuple[str, ...]): The data the model is built from (the names)
        model (torch.Tensor): The model fitting the data (a matrix)

    Returns:
        float: The average negative log likelihood score
    """
    # As the matrix model is using bigrams, we convert the data to bigrams
    log_likelihood = 0.0
    n = 0

    for name in data:
        for token_1, token_2 in zip(name, name[1:]):
            idx1 = ALPHABET_DICT[token_1]
            idx2 = ALPHABET_DICT[token_2]
            probability = model[idx1, idx2]
            log_probability = probability.log().item()
            # Multiplying the probabilities is equivalent of summing the log of
            # the probabilities
            log_likelihood += log_probability
            n += 1

    avg_nll = -log_likelihood / n
    return avg_nll


if __name__ == "__main__":
    from makemore_bigram.preprocessing import get_padded_data, get_probability_matrix

    names_ = get_padded_data()
    probability_matrix_ = get_probability_matrix()
    avg_nll_ = calculate_avg_nll_of_matrix_model(names_, probability_matrix_)
    print(avg_nll_)
