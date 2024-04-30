"""Module to run inference on the model."""

import argparse
import sys
from typing import List, Tuple

import torch
from makemore_agb.data_classes import ModelParams, OptimizationParams
from makemore_agb.predict import predict_neural_network
from makemore_agb.train import train

from makemore_agb import INDEX_TO_TOKEN, TOKEN_TO_INDEX


def run_inference(
    model: Tuple[torch.Tensor, ...],
    n_samples: int = 20,
    seed: int = 2147483647,
) -> Tuple[str, ...]:
    """Run inference on the model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model to run inference on.
        n_samples (int, optional): The number of inferences to run.
            Defaults to 20.
        seed (int, optional): The seed to use. Defaults to 2147483647.

    Returns:
        Tuple[str, ...]: The predictions
    """
    # Obtain the embedding size from c
    embedding_size = int(model[0].shape[-1])
    # Obtain the block size from w1
    block_size = int(model[1].shape[-2] / embedding_size)

    g = torch.Generator().manual_seed(seed)
    predictions: List[str] = []

    for _ in range(n_samples):
        characters = ""
        context = [TOKEN_TO_INDEX["."]] * block_size  # Initialize with stop characters

        while True:
            # Note the [] to get the batch shape correct
            logits = predict_neural_network(
                model=model, input_data=torch.tensor([context])
            )
            probs = torch.softmax(logits, dim=1)
            index = torch.multinomial(probs, num_samples=1, generator=g)
            # The context size is constant, so we drop the first token, and add
            # the predicted token to the next
            context = context[1:] + [index]
            if index == 0:
                break
            characters += f"{INDEX_TO_TOKEN[index.item()]}"

        predictions.append(characters)

    return tuple(predictions)


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    """Parse the arguments.

    Args:
        sys_args (List[str]): The system arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Predict on the MLP model.",
    )

    parser.add_argument(
        "-n",
        "--n-predictions",
        type=int,
        required=False,
        default=20,
        help=("Number of names to predict"),
    )

    args = parser.parse_args(sys_args)
    return args


def main(sys_args: List[str]):
    """Parse the arguments and run train_and_plot.

    Args:
        sys_args (List[str]): The system arguments
    """
    args = parse_args(sys_args)
    model_params = ModelParams(
        embedding_size=10,
        hidden_layer_neurons=200,
    )
    optimization_params = OptimizationParams(
        n_mini_batches=200_000,
        mini_batches_per_data_capture=1_000,
    )
    model, _ = train(model_params=model_params, optimization_params=optimization_params)
    predictions = run_inference(model=model, n_samples=args.n_predictions)
    for prediction in predictions:
        print(f"{prediction}")


if __name__ == "__main__":
    main(sys.argv[1:])
