"""Module for inference."""

from typing import List, Literal, Tuple

import torch
from makemore_bigram.train import get_neural_net, get_probability_matrix

from makemore_bigram import INDEX_TO_TOKEN


def sample_from_matrix(
    probability_matrix: torch.Tensor, n_samples: int = 20, seed: int = 2147483647
) -> Tuple[str, ...]:
    """Draw samples from the probability matrix.

    NOTE: Normally one would predict from some input data, here however, we are
          sampling from the matrix which is the model

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
            sample.append(INDEX_TO_TOKEN[idx])
        samples.append("".join(sample))
    return tuple(samples)


def main(model: Literal["probability_matrix", "neural_net"]) -> None:
    """Train and run inference on model.

    Args:
        model (Literal['probability_matrix', 'neural_net']): The model to run
            inference on
    """
    if model == "probability_matrix":
        matrix = get_probability_matrix()
    elif model == "neural_net":
        matrix = get_neural_net()

    samples = sample_from_matrix(probability_matrix=matrix)
    for sample in samples:
        print(sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=("probability_matrix", "neural_net"),
        help=(
            "What model to use for inference. "
            "The probability matrix is trained through counts. "
            "The neural net will be trained through backpropagation."
        ),
    )
    args = parser.parse_args()

    main(args.model)
