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


if __name__ == "__main__":
    from makemore_bigram.preprocessing import get_probability_matrix

    probability_matrix_ = get_probability_matrix()
    samples_ = sample_from_matrix(probability_matrix=probability_matrix_)
    for sample_ in samples_:
        print(sample_)
