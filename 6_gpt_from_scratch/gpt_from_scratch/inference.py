"""Module to run inference on the model."""

import argparse
import sys
from typing import List, Tuple

import torch
from gpt_from_scratch.data_classes import ModelParams, OptimizationParams, ModelType
from gpt_from_scratch.module import Sequential
from gpt_from_scratch.train import train

from gpt_from_scratch import DEVICE, INDEX_TO_TOKEN, TOKEN_TO_INDEX


def run_inference(
    model: Sequential,
    model_params: ModelParams,
    n_samples: int = 20,
    seed: int = 2147483647,
) -> Tuple[str, ...]:
    """Run inference on the model.

    Args:
        model (Sequential): The model to run inference on.
        model_params (ModelParams): The parameters of the model
        n_samples (int, optional): The number of inferences to run.
            Defaults to 20.
        seed (int, optional): The seed to use. Defaults to 2147483647.

    Returns:
        Tuple[str, ...]: The predictions
    """
    # Disable training
    for layer in model.layers:
        if hasattr(layer, "training"):
            layer.training = False

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    predictions: List[str] = []

    for _ in range(n_samples):
        characters = ""
        # Initialize with stop characters
        context = [TOKEN_TO_INDEX["."]] * model_params.block_size

        while True:
            # Note the [] to get the batch shape correct
            logits = model(torch.tensor([context]))
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
    parser.add_argument(
        "-t",
        "--model-type",
        type=ModelType,
        required=False,
        default=ModelType.NONE,
        choices=list(ModelType),
        help=(
            "The pre-defined models. "
            "If selected, they will overwrite other parameters set in the input."
        ),
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

    model, _ = train(
        model_params=model_params,
        optimization_params=optimization_params,
        model_type=args.model_type,
    )
    predictions = run_inference(
        model=model,
        model_params=model_params,
        n_samples=args.n_predictions,
    )
    for prediction in predictions:
        print(f"{prediction}")


if __name__ == "__main__":
    main(sys.argv[1:])
