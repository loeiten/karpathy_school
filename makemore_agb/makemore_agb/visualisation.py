"""Module for visualisation."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from makemore_agb.data_classes import LayerType, TrainStatistics
from makemore_agb.linear import Linear
from makemore_agb.module import Module
from matplotlib.axes import Axes

# Use nice theme when plotting
sns.set_theme()


def plot_training(train_statistics: TrainStatistics):
    """
    Plot training data.

    Args:
        train_statistics (TrainStatistics): The statistics of the train job
    """
    _, ax = plt.subplots()
    ax.plot(
        train_statistics.training_step,
        np.log(train_statistics.training_loss),
        label="(Batch wise) training loss",
    )
    ax.plot(
        train_statistics.eval_training_step,
        np.log(train_statistics.eval_training_loss),
        label="Training loss",
    )
    ax.plot(
        train_statistics.eval_validation_step,
        np.log(train_statistics.eval_validation_loss),
        label="Validation loss",
    )
    ax.set_ylabel("Log(loss)")
    ax.set_xlabel("Step")
    ax.legend(loc="best", fancybox=True)
    ax.set_title("Training loss")
    plt.show()


def plot_histogram(tensor: torch.Tensor, tensor_name: str, ax: Axes) -> None:
    """
    Plot a histogram of the values of the tensor.

    Args:
        tensor (torch.Tensor): The tensor to plot
        tensor_name (str): Name of the tensor
        ax (Axes): The axes to plot on
    """
    ax.hist(tensor.view(-1).tolist(), bins=50)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value")
    ax.set_title(f"Histogram of {tensor_name}")


def plot_dead_neuron(
    tensor: torch.Tensor, tensor_name: str, ax: Axes, threshold: float = 0.99
) -> None:
    """
    Plot dead neurons for a given batch of examples.

    White denotes dead neurons in the plot

    Args:
        tensor (torch.Tensor): The tensor to plot
        tensor_name (str): Name of the tensor
        ax (Axes): The axes to plot on
        threshold (float): The threshold of which we will call a neuron dead
    """
    ax.imshow(tensor.abs() > threshold, cmap="gray", interpolation="nearest")
    ax.set_ylabel("Example number")
    ax.set_xlabel("Neuron number")
    ax.grid(False)
    ax.set_title(f"Dead neurons of {tensor_name}")


def plot_activation_distribution_per_layer(
    model: Tuple[Module], ax: Axes, layer_type: LayerType, use_gradients: bool = False
) -> None:
    """Plot and report the distribution of the activation functions.

    Note:
    - If we don't have batch normalization
      - And we have correct gain: Gradients looks good, but layer 1 has several
        saturated neurons which will block the learning due to flat gradients in
        this region
      - Too small gain: Std dev is not maintained to be 1 anymore, it decreases
        for each layer because of tanh (flattened distributions gives more
        extreme values for tanh which gives bad learning)
      - Too small gain and no tanh: We won't have non-linearities, so the
        network only learns linear relationships, we also see that activations
        and learning effects decreases the deeper we go into the network
    - If we have batch normalization we don't have to do the balancing act of
      setting the correct initialization

    Args:
        model (Tuple[Module]): The model to plot the activations from
        ax (Axes): The axes to plot on
        layer_type (LayerType): The layer type to plot
        use_gradients (bool, optional): Will use gradients instead of weights.
            Defaults to False
    """
    print(
        f"Report for {'activations' if not use_gradients else 'gradients'} of "
        f"{layer_type.value}"
    )
    for layer_nr, layer in enumerate(model):
        if layer.__class__.__name__ == layer_type.value:  # type: ignore
            tensor = layer.out.grad if use_gradients else layer.out
            hy, hx = torch.histogram(tensor, density=True)
            ax.plot(
                hx[:-1].detach(),
                hy.detach(),
                label=f"Layer {layer_nr} ({layer.__class__.__name__})",
            )
            additional_string = ""
            if not use_gradients:
                additional_string = (
                    f", Saturated: {(tensor.abs() > 0.97).float().mean()*100:.2f} %"
                )
            if layer_type == LayerType.LINEAR:
                additional_string = (
                    ", gradient/weight std ratio: "
                    f"{layer.out.grad.std()/layer.out.std():.2e}"
                )
            print(
                f"Layer {layer_nr} ({layer.__class__.__name__}): "
                f"Mean: {tensor.mean():+.2f}, "
                f"Std: {tensor.std():.2f}"
                f"{additional_string}"
            )

    ax.set_title(
        f"{layer_type.value} "
        f"{'gradient' if use_gradients else 'activation'} "
        "distribution"
    )
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value")
    ax.legend(loc="best", fancybox=True)


def plot_update_to_data_ratio(
    model: Tuple[Module], update_ratio: List[List[float]], ax: Axes
) -> None:
    """Plot the update-to-data ratio.

    Args:
        model (Tuple[Module]): The model to plot the activations from
        update_ratio (List[List[float]]): The update ratio
        ax (Axes): The axes to plot on
    """
    for layer_nr, layer in enumerate(model):
        if isinstance(layer, Linear):
            ax.plot(
                [update_ratio[i][layer_nr] for i in range(len(update_ratio))],
                label=f"Layer {layer_nr} ({layer.__class__.__name__})",
            )

    ax.plot([0, len(update_ratio)], [-3, -3], "k", label=r"$10^{-3}$")

    ax.set_title("Update to data ratio")
    ax.set_ylabel(r"$\log_{10}\frac{\sigma_{LR\cdot \nabla}}{W}$")
    ax.set_xlabel("Iteration")
    ax.legend(loc="best", fancybox=True)
