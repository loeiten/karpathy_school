"""
Module containing metric calculation.

About negative log likelihood (nll):

If the model predicts all the bigrams in the data set perfectly, then it will
assign the probability 1 to all.

However, our models are only seeing bigrams, and have no notion of where in the
name this bigram occurs.
We can loop through the probabilities of the model (aka the matrix) assigns to
each bigram occurring in the data set.
If we want one number describing the quality of the model, we can multiply these
numbers to get what is called the likelihood.
As the probabilities are less than 1, this multiplication results in a very
small number which can lead to high truncation errors.
Hence, we take the logarithm of this number, and get the log likelihood.

If we want a good model we can maximize the likelihood.
This is the same as maximizing the log likelihood as log is monotonically
increasing.
This is the same as minimizing the negative of the log likelihood (nll), which
is appropriate for gradient DESCENT.
In order for the loss not to get to high, it's common to normalize the nll.

A model with average nll = 0 is a model which perfectly fits the data.
"""

from typing import Tuple

import torch

from makemore_bigram import TOKEN_TO_INDEX


def calculate_avg_nll_of_probability_matrix_model(
    data: Tuple[str, ...], probability_matrix: torch.Tensor
) -> torch.Tensor:
    """Calculate the average nll of the probability matrix model.

    NOTE: Normally we compare a prediction against the ground truth.
          However, as the matrix contains all the probability for all bigrams it
          contains all the information needed to calculate the average nll

    Args:
        data (Tuple[str, ...]): The data the model is built from (the names)
        probability_matrix (torch.Tensor): The model fitting the data

    Returns:
        torch.Tensor: The average negative log likelihood score
    """
    log_likelihood = 0.0
    n = 0

    # As the matrix model is using bigrams, we convert the data to bigrams
    for name in data:
        for token_1, token_2 in zip(name, name[1:]):
            idx1 = TOKEN_TO_INDEX[token_1]
            idx2 = TOKEN_TO_INDEX[token_2]
            probability = probability_matrix[idx1, idx2]
            log_probability = probability.log()
            # Multiplying the probabilities is equivalent of summing the log of
            # the probabilities
            log_likelihood += log_probability
            n += 1

    avg_nll = -log_likelihood / n
    return avg_nll


def calculate_avg_nll_of_neural_net_model(
    probabilities: torch.Tensor, ground_truth: torch.Tensor, n_examples: int
) -> torch.Tensor:
    """Calculate the avg nll of the neural net model.

    Args:
        probabilities (torch.Tensor): The probabilities obtained by running
            inference on the entire data-set
        ground_truth (torch.Tensor): The token which follows the input token in
            the data set
        n_examples (int): Number of examples

    Returns:
        torch.Tensor: The average negative log likelihood score
    """
    # NOTE: This is not the same as probabilities[:, ground_truth]
    #       In the first indexing the input is [0, 1, 2, ..., n_examples-1].
    #       This selects all the rows
    #       The second indexing selects which element to pick from the row
    #       selected by the first index
    #       Hence the output shape of
    #       probabilities[torch.arange(n_examples), ground_truth] will be
    #       [n_elements]
    return -probabilities[torch.arange(n_examples), ground_truth].log().mean()
