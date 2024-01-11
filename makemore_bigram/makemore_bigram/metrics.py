"""Module containing metric calculation."""

from typing import Tuple

import torch

from makemore_bigram import TOKEN_TO_INDEX


def calculate_avg_nll_of_matrix_model(
    data: Tuple[str, ...], model: torch.Tensor
) -> torch.Tensor:
    """Calculate the average negative log likelihood (nll) of the matrix model.

    If the model predicts all the bigrams in the data set perfectly, then it
    will assign the probability 1 to all.

    However, our models are only seeing bigrams, and have no notion of where in
    the name this bigram occurs.
    We can loop through the probabilities of the model (the matrix) assigns to
    each bigram occurring in the data set.
    If we want one number describing the quality of the model, we can multiply
    these numbers to get what is called the likelihood.
    As the probabilities are less than 1, this multiplication results in a very
    small number which can lead to high truncation errors.
    Hence, we take the logarithm of this number, and get the log likelihood.

    If we want a good model we can maximize the likelihood.
    This is the same as maximizing the log likelihood as log is monotonically
    increasing.
    This is the same as minimizing the negative of the log likelihood (nll),
    which is appropriate for gradient DESCENT.
    In order for the loss not to get to high, it's common to normalize the nll.

    A model with average nll = 0 is a model which perfectly fits the data

    NOTE: Normally we compare a prediction against the ground truth. However, as
    the matrix contains all the probability for all bigrams it contains all the
    information needed to calculate the average nll

    Args:
        data (Tuple[str, ...]): The data the model is built from (the names)
        model (torch.Tensor): The model fitting the data (a matrix)

    Returns:
        torch.Tensor: The average negative log likelihood score
    """
    # As the matrix model is using bigrams, we convert the data to bigrams
    log_likelihood = 0.0
    n = 0

    for name in data:
        for token_1, token_2 in zip(name, name[1:]):
            idx1 = TOKEN_TO_INDEX[token_1]
            idx2 = TOKEN_TO_INDEX[token_2]
            probability = model[idx1, idx2]
            log_probability = probability.log()
            # Multiplying the probabilities is equivalent of summing the log of
            # the probabilities
            log_likelihood += log_probability
            n += 1

    avg_nll = -log_likelihood / n
    return avg_nll
