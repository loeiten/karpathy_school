"""Module for visualisation."""

from typing import Tuple

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


def plot_activation_distribution_per_layer(
    model: Tuple[Module], ax: Axes, use_gradients: bool = False
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
        use_gradients (bool, optional): Will use gradients instead of weights.
            Defaults to False
    """
    print(f"Report for {'activations' if not use_gradients else 'gradients'}")
    for layer_nr, layer in enumerate(model):
        if isinstance(layer, Tanh):
            tensor = layer.out.grad if use_gradients else layer.out
            hy, hx = torch.histogram(tensor, density=True)
            ax.plot(
                hx[:-1].detach(),
                hy.detach(),
                label=f"Layer {layer_nr} ({layer.__class__.__name__})",
            )
            saturation_string = ""
            if not use_gradients:
                saturation_string = (
                    f", Saturated: {(tensor.abs() > 0.97).float().mean()*100:.2f} %"
                )
            print(
                f"Layer {layer_nr} ({layer.__class__.__name__}): "
                f"Mean: {tensor.mean():+.2f}, "
                f"Std: {tensor.std():.2f}"
                f"{saturation_string}"
            )

    ax.set_title(f"{'Gradient' if use_gradients else 'Activation'} distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value")
    ax.legend(loc="best", fancybox=True)
