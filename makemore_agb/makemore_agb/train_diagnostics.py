"""Module to diagnose distribution."""

import argparse
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
from makemore_agb.data_classes import (
    LayerType,
    ModelParams,
    OptimizationParams,
    TrainStatistics,
)
from makemore_agb.models import get_model_function
from makemore_agb.module import Module
from makemore_agb.preprocessing import get_dataset
from makemore_agb.train import train_neural_net_model
from makemore_agb.visualisation import plot_activation_distribution_per_layer


def plot_pytorch_train_distributions(
    good_initialization: bool = False,
    batch_normalize: bool = False,
    seed: int = 2147483647,
    show=True,
) -> None:
    """Plot the training distribution of the pytorch model after 1000 steps.

    Args:
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights
        batch_normalize (bool): Whether or not to use batch normalization
        seed (int): The seed for the random number generator
        show (bool): Whether or not to show the plot
    """
    model_params = ModelParams(
        block_size=3,
        embedding_size=10,
        hidden_layer_neurons=100,
        seed=seed,
        good_initialization=good_initialization,
        batch_normalize=batch_normalize,
    )

    model_function = get_model_function("pytorch")

    model = model_function(model_params)
    dataset = get_dataset(block_size=model_params.block_size)

    _ = train_neural_net_model(
        model_type="pytorch",
        model=model,
        dataset=dataset,
        optimization_params=OptimizationParams(
            n_mini_batches=1000, mini_batches_per_data_capture=1000, batch_size=32
        ),
        train_statistics=TrainStatistics(),
        batch_normalization_parameters=None,
    )
    # We know that model must be of type Tuple[Module]
    plot_distributions_from_pytorch_model(model=model, show=show)  # type: ignore


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
        [
            ["tanh_activations"],
            ["tanh_gradients"],
            ["linear_gradients"],
            ["update_to_data_ratio"],
        ],
        layout="constrained",
    )

    plot_activation_distribution_per_layer(
        model=model,
        ax=axes["tanh_activations"],
        layer_type=LayerType.TANH,
        use_gradients=False,
    )
    plot_activation_distribution_per_layer(
        model=model,
        ax=axes["tanh_gradients"],
        layer_type=LayerType.TANH,
        use_gradients=True,
    )
    plot_activation_distribution_per_layer(
        model=model,
        ax=axes["linear_gradients"],
        layer_type=LayerType.LINEAR,
        use_gradients=True,
    )

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

    args = parser.parse_args(sys_args)
    return args


def main(sys_args: List[str]):
    """Parse the arguments and run plot_initial_distributions.

    Args:
        sys_args (List[str]): The system arguments
    """
    args = parse_args(sys_args)
    plot_pytorch_train_distributions(
        good_initialization=args.good_initialization,
        batch_normalize=args.batch_normalize,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
