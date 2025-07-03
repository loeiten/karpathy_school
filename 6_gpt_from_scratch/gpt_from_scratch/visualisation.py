"""Module for visualisation."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gpt_from_scratch.data_classes import TrainStatistics

# Use nice theme when plotting
sns.set_theme()


def plot_training(train_statistics: TrainStatistics, average_window: int = 1000):
    """
    Plot training data.

    Args:
        train_statistics (TrainStatistics): The statistics of the train job
        average_window (int): The number of samples to take the average over
            when plotting the training_loss
    """
    _, ax = plt.subplots()
    training_step = torch.tensor(train_statistics.training_step).view(
        -1, average_window
    )[:, 0]
    training_loss = (
        torch.tensor(train_statistics.training_loss).view(-1, average_window).mean(1)
    )
    ax.plot(
        training_step,
        np.log(training_loss),
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
