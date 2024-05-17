"""Module to diagnose distribution."""

import argparse
import sys
from typing import List

import matplotlib.pyplot as plt
import torch
from makemore_agb.models import get_model
from makemore_agb.predict import predict_neural_network
from makemore_agb.preprocessing import get_dataset
from makemore_agb.visualisation import plot_dead_neuron, plot_histogram


def plot_initial_distributions(
    good_initialization: bool = False,
    batch_normalize: bool = False,
    seed: int = 2147483647,
    show=True,
) -> None:
    """Plot the initial distribution.

    Raises:
        RuntimeError: In case the prediction outputs an output with unexpected
            length

    Args:
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights
        batch_normalize (bool): Whether or not to use batch normalization
        seed (int): The seed for the random number generator
        show (bool): Whether or not to show the plot
    """
    block_size = 3
    batch_size = 32
    g = torch.Generator().manual_seed(seed)
    model = get_model(
        block_size=block_size,
        embedding_size=10,
        hidden_layer_neurons=200,
        good_initialization=good_initialization,
    )
    dataset = get_dataset(block_size=block_size)
    training_data = dataset["training_input_data"]
    idxs = torch.randint(
        low=0, high=training_data.shape[0], size=(batch_size,), generator=g
    )
    output = predict_neural_network(
        model=model,
        input_data=training_data[idxs],
        batch_normalize=batch_normalize,
        inspect_pre_activation_and_h=True,
    )
    if len(output) != 3:
        raise RuntimeError("Got unexpected output from the predictor")

    # We're checking for the length above, so we can safely ignore the pylint
    _, h_pre_activation, h = output  # pylint: disable=unbalanced-tuple-unpacking

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
        default=False,
        help=(
            "Whether or not to use an initialization which has a good "
            "distribution of the initial weights"
        ),
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--batch-normalize",
        required=False,
        help=("Whether or not to batch normalization"),
        action="store_true",
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
        good_initialization=args.good_initialization,
        batch_normalize=args.batch_normalize,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
