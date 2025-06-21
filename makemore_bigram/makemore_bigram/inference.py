"""Module for inference."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from makemore_bigram.train import get_neural_net, get_probability_matrix
from makemore_bigram.utils.inference_helper import (
    neural_net_inference_without_interpretation,
)

from makemore_bigram import INDEX_TO_TOKEN, N_TOKENS, ModelTypes


def run_inference(
    model_type: ModelTypes,
    matrix: torch.Tensor,
    n_samples: int = 20,
    seed: int = 2147483647,
) -> Tuple[str, ...]:
    """Draw samples from the probability matrix.

    NOTE: Normally one would predict from some input data, here however, we are
          sampling from the matrix which is the model

    Args:
        model_type (MODEL_TYPE): Model type to run
        matrix (torch.Tensor): If model_type=='probability_matrix': Matrix
            containing probability of the next sample.
            If model_type=='neural_net': The weights of the neural net
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
            if model_type == "probability_matrix":
                # If we select the first token, the columns will be the
                # probabilities for the following token
                probability_matrix = matrix[idx, :]
            elif model_type == "neural_net":
                # Here we first select a one hot encoded token
                # The following will create a 1xN_TOKENS tensor
                input_data = F.one_hot(
                    torch.tensor([idx]), num_classes=N_TOKENS
                ).float()
                # We multiply the encoded token with the model weights in order
                # to get the prediction
                probability_matrix = neural_net_inference_without_interpretation(
                    input_data=input_data, weights=matrix
                )
            idx = torch.multinomial(
                probability_matrix, num_samples=1, replacement=True, generator=g
            ).item()
            # Stop when stop token is reached
            if idx == 0:
                break
            sample.append(INDEX_TO_TOKEN[idx])
        samples.append("".join(sample))
    return tuple(samples)


def main(model_type: ModelTypes) -> None:
    """Train and run inference on model.

    Args:
        model_type (ModelTypes): The model type to run inference on
    """
    if model_type == "probability_matrix":
        matrix = get_probability_matrix()
    elif model_type == "neural_net":
        matrix = get_neural_net()

    samples = run_inference(model_type=model_type, matrix=matrix)
    for sample in samples:
        print(sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=ModelTypes.__args__,  # type: ignore
        help=(
            "What model to use for inference. "
            "The probability matrix is trained through counts. "
            "The neural net will be trained through backpropagation."
        ),
    )
    args = parser.parse_args()

    main(args.model)
