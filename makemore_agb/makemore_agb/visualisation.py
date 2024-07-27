"""Module for visualisation."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from makemore_agb.data_classes import TrainStatistics
from makemore_agb.module import Module
from makemore_agb.tanh import Tanh
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


def plot_activation_distribution_per_layer(model: List[Module], ax: Axes) -> None:
    """Plot and report the distribution of the activation functions.

    Args:
        model (List[Module]): The model to plot the activations from
        ax (Axes): The axes to plot on
    """
    for layer_nr, layer in enumerate(model):
        if isinstance(layer, Tanh):
            out = layer.out
            hy, hx = torch.histogram(out, density=True)
            ax.plot(
                hx,
                hy,
                label=f"Layer {layer_nr} ({layer.__class__.__name__})",
            )
            print(
                f"Layer {layer_nr} ({layer.__class__.__name__}): "
                f"Mean: {out.mean():+.2f}, "
                f"Std: {out.std():.2f}, "
                f"Saturated: {(out.abs() > 0.97).abs().mean()*100:.2f} %"
            )

    ax.set_title("Activation distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value")
    ax.legend(loc="best", fancybox=True)
