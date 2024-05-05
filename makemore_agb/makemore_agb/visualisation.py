"""Module for visualisation."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from makemore_agb.data_classes import TrainStatistics
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
