"""Module to diagnose distribution."""

import argparse
import sys
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import torch
from makemore_agb.data_classes import (
    BatchNormalizationParameters,
    ModelParams,
    OptimizationParams,
)
from makemore_agb.models import get_model_function
from makemore_agb.module import Module
from makemore_agb.predict import predict_neural_network
from makemore_agb.preprocessing import get_dataset
from makemore_agb.train import train_neural_net_model
from makemore_agb.visualisation import (
    plot_activation_distribution_per_layer,
    plot_dead_neuron,
    plot_histogram,
)

from makemore_agb import DEVICE


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-locals
def plot_initial_distributions(
    model_type: Literal["explicit", "pytorch"],
    good_initialization: bool = False,
    batch_normalize: bool = False,
    seed: int = 2147483647,
    show=True,
) -> None:
    """Plot the initial distribution.

    Args:
        model_type (Literal["explicit", "pytorch"]): What model type to use
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights
        batch_normalize (bool): Whether or not to use batch normalization
        seed (int): The seed for the random number generator
        show (bool): Whether or not to show the plot

    Raises:
        ValueError: If unknown model_type is given
    """
    model_params = ModelParams(
        block_size=3,
        embedding_size=10,
        hidden_layer_neurons=200 if model_type == "explicit" else 100,
        seed=seed,
        good_initialization=good_initialization,
        batch_normalize=batch_normalize,
    )
    batch_size = 32
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    model_function = get_model_function(model_type)

    model = model_function(model_params)
    dataset = get_dataset(block_size=model_params.block_size)
    training_data = dataset["training_input_data"]
    idxs = torch.randint(
        low=0,
        high=training_data.shape[0],
        size=(batch_size,),
        generator=g,
        device=DEVICE,
    )
    if batch_normalize and model_type == "explicit":
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

    if model_type == "explicit":
        output = predict_neural_network(
            model_type=model_type,
            model=model,
            input_data=training_data[idxs],
            batch_normalization_parameters=batch_normalization_parameters,
            inspect_pre_activation_and_h=True,
        )
        plot_distributions_from_explicit_model(output=output, show=show)
    elif model_type == "pytorch":
        _ = train_neural_net_model(
            model_type=model_type,
            model=model,
            dataset=dataset,
            optimization_params=OptimizationParams(
                n_mini_batches=1, mini_batches_per_data_capture=1, batch_size=32
            ),
            train_statistics=None,
            batch_normalization_parameters=None,
        )
        # We know that model must be of type Tuple[Module]
        plot_distributions_from_pytorch_model(model=model, show=show)  # type: ignore
    else:
        raise ValueError(f"Unknown model_type {model_type}")


def plot_distributions_from_explicit_model(
    output: torch.Tensor, show: bool = True
) -> None:
    """Plot the distributions from the explicit model.

    Args:
        output (torch.Tensor): The output from a prediction, must be of size 3
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Raises:
        RuntimeError: If unexpected input is obtained.
    """
    if len(output) != 3:
        raise RuntimeError("Got unexpected output from the predictor")

    # We're checking for the length above, so we can safely ignore the pylint
    # pylint: disable-next=unbalanced-tuple-unpacking
    _, h_pre_activation, h = output

    # Create the figures
    _, axes = plt.subplot_mosaic(
        [["h_pre_activation", "h"], ["dead_neurons", "dead_neurons"]],
        layout="constrained",
    )

    plot_histogram(
        tensor=h_pre_activation,
        tensor_name="h pre-activation",
        ax=axes["h_pre_activation"],
    )
    plot_histogram(tensor=h, tensor_name="h", ax=axes["h"])
    plot_dead_neuron(tensor=h, tensor_name="h", ax=axes["dead_neurons"], threshold=0.99)

    if show:
        plt.show()


def plot_distributions_from_pytorch_model(
    model: Tuple[Module], show: bool = True
) -> None:
    """Plot the distributions from the explicit model.

    Args:
        model (Tuple[Module]): The model
        show (bool, optional): Whether or not to show the plot. Defaults to True.
    """
    # Create the figures
    _, axes = plt.subplot_mosaic(
        [["activations"]],
        layout="constrained",
    )

    plot_activation_distribution_per_layer(model=model, ax=axes["activations"])

    if show:
        plt.show()


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    """Parse the arguments.

    Args:
        sys_args (List[str]): The system arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Plot the initial distribution diagnostics.",
    )

    parser.add_argument(
        "-g",
        "--good-initialization",
        help=(
            "Whether or not to use an initialization which has a good "
            "distribution of the initial weights"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--batch-normalize",
        help=("Whether or not to batch normalization"),
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
    """Parse the arguments and run plot_initial_distributions.

    Args:
        sys_args (List[str]): The system arguments
    """
    args = parse_args(sys_args)
    plot_initial_distributions(
        model_type=args.model_type,
        good_initialization=args.good_initialization,
        batch_normalize=args.batch_normalize,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
