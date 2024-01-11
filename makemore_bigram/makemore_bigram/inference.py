"""Module for inference."""

from typing import List, Tuple

import torch

from makemore_bigram import INVERSE_ALPHABET_DICT


def sample_from_matrix(
    probability_matrix: torch.Tensor, n_samples: int = 20, seed: int = 2147483647
) -> Tuple[str, ...]:
    """Draw samples from the probability matrix.

    Args:
        probability_matrix (torch.Tensor): Matrix containing probability of the
            next sample
        n_samples (int): Number of samples to draw
        seed (int): The seed to the generator

    Returns:
        Tuple[str, ...]: Samples drawn
    """
    g = torch.Generator().manual_seed(seed)

    samples: List[str] = []
    for _ in range(n_samples):
        sample: List[str] = []
        # Start with start/stop token
        idx = 0
        while True:
            idx = torch.multinomial(
                probability_matrix[idx, :], num_samples=1, replacement=True, generator=g
            ).item()
            # Stop when stop token is reached
            if idx == 0:
                break
            sample.append(INVERSE_ALPHABET_DICT[idx])
        samples.append("".join(sample))
    return tuple(samples)


def neural_net_inference_without_interpretation(
    input_data: torch, weights: torch.Tensor
) -> torch.Tensor:
    """Run inference on the neural net without interpretation of the data.

    Args:
        input_data (torch.Tensor): The data to run the model on
        weights (torch.Tensor): The model
            (one hidden layer without bias, i.e. a matrix)

    Returns:
        torch.Tensor: The predictions
    """
    # Predict the log counts
    logits = input_data @ weights
    # This is equivalent to the count matrix
    counts = logits.exp()
    # NOTE: This is equivalent to softmax
    probabilities = counts / counts.sum(dim=1, keepdim=True)
    return probabilities


def neural_net_inference_with_interpretation(
    input_data: torch, weights: torch.Tensor, n_predictions: int = 10
) -> Tuple[str, ...]:
    """Run inference with interpretation.

    Args:
        input_data (torch): The data to run predictions on
        weights (torch.Tensor): The model (a.k.a. the weights)
        n_predictions (int): Number of predictions to make

    Returns:
        Tuple[str, ...]: The interpreted predictions
    """
    # FIXME: Implement this
    print(f"{input_data=}, {weights=}, {n_predictions=}")
    return tuple(["foo", "bar"])


if __name__ == "__main__":
    from makemore_bigram.models import get_matrix_model

    probability_matrix_ = get_matrix_model()
    samples_ = sample_from_matrix(probability_matrix=probability_matrix_)
    for sample_ in samples_:
        print(sample_)
