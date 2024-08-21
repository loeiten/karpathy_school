"""Module to run inference on the model."""

import argparse
import sys
from typing import List, Literal, Optional, Tuple, Union

import torch
from makemore_agb.data_classes import (
    BatchNormalizationParameters,
    ModelParams,
    OptimizationParams,
)
from makemore_agb.embedding import Embedding
from makemore_agb.linear import Linear
from makemore_agb.module import Module
from makemore_agb.predict import predict_neural_network
from makemore_agb.train import train

from makemore_agb import DEVICE, INDEX_TO_TOKEN, TOKEN_TO_INDEX


def run_inference(
    model_type: Literal["explicit", "pytorch"],
    model: Union[Tuple[torch.Tensor, ...], Tuple[Module, ...]],
    n_samples: int = 20,
    batch_normalization_parameters: Optional[BatchNormalizationParameters] = None,
    seed: int = 2147483647,
) -> Tuple[str, ...]:
    """Run inference on the model.

    Args:
        model_type (Literal["explicit", "pytorch"]): What model type to use
        model (Union[Tuple[torch.Tensor, ...], Tuple[Module, ...]]): The model to
            run inference on.
        n_samples (int, optional): The number of inferences to run.
            Defaults to 20.
        batch_normalization_parameters (Optional[BatchNormalizationParameters]):
            If set: Contains the running mean and the running standard deviation
        seed (int, optional): The seed to use. Defaults to 2147483647.

    Raises:
        TypeError: If invalid model is given

    Returns:
        Tuple[str, ...]: The predictions
    """
    if (
        model_type == "explicit"
        and isinstance(model[0], torch.Tensor)
        and isinstance(model[1], torch.Tensor)
    ):
        # Obtain the embedding size from c
        embedding_size = int(model[0].shape[-1])
        # Obtain the block size from w1
        block_size = int(model[1].shape[-2] / embedding_size)
    elif (
        model_type == "pytorch"
        and isinstance(model[0], Embedding)
        and isinstance(model[1], Linear)
    ):
        # Obtain the embedding size from c
        embedding_size = int(model[0].weight.shape[-1])
        # Obtain the block size from w1
        block_size = int(model[1].weight.shape[-2] / embedding_size)
        # Disable training
        for layer in model:
            if hasattr(layer, "training"):
                layer.training = False
    else:
        raise TypeError(
            f"{model_type=} with {type(model[0])=} and {type(model[1])=} not recognized"
        )

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    predictions: List[str] = []

    for _ in range(n_samples):
        characters = ""
        context = [TOKEN_TO_INDEX["."]] * block_size  # Initialize with stop characters

        while True:
            # Note the [] to get the batch shape correct
            # Note the [0] as predict always returns a tuple
            logits = predict_neural_network(
                model_type=model_type,
                model=model,
                input_data=torch.tensor([context]),
                batch_normalization_parameters=batch_normalization_parameters,
                training=False,
            )[0]
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
        "-m",
        "--batch-normalize",
        help=("Whether or not to use batch normalization"),
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--model-type",
        type=str,
        choices=("explicit", "pytorch"),
        help="What model type to use",
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
    if args.batch_normalize:
        # These parameters will be used as batch norm parameters during inference
        # Initialized to zero as the mean and one as std as the initialization of w1
        # and b1 is so that h_pre_activation is roughly gaussian
        batch_normalization_parameters = BatchNormalizationParameters(
            running_mean=torch.zeros(
                (1, model_params.hidden_layer_neurons),
                requires_grad=False,
                device=DEVICE,
            ),
            running_std=torch.ones(
                (1, model_params.hidden_layer_neurons),
                requires_grad=False,
                device=DEVICE,
            ),
        )
    else:
        batch_normalization_parameters = None
    model, _ = train(
        model_type=args.model_type,
        model_params=model_params,
        optimization_params=optimization_params,
        batch_normalization_parameters=batch_normalization_parameters,
    )
    predictions = run_inference(
        model_type=args.model_type,
        model=model,
        n_samples=args.n_predictions,
        batch_normalization_parameters=batch_normalization_parameters,
    )
    for prediction in predictions:
        print(f"{prediction}")


if __name__ == "__main__":
    main(sys.argv[1:])
