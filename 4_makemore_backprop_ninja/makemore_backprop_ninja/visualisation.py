"""Module for visualisation."""

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Use nice theme when plotting
sns.set_theme()


def plot_dl_d_logits(dl_d_logits: torch.Tensor):
    """
    Plot dl/dlogits.

    dl_d_logits[x].sum() will be equal to 0
    The black spots are the correct indices (where we subtracted with 1)
    I.e. per row (batch) we are pulling up the probabilities of the correct
    index, and we're pulling down all the others.
    The amount by which your prediction is incorrect is exactly the amount by
    which you get a pull or push in that dimension.
    A confidently mis-predicted element will be pulled down more heavily.
    The gradient is proportional to the degree of mis-prediction.

    Args:
        dl_d_logits (torch.Tensor): The derivative of l w.r.t. the logits
    """
    _, ax = plt.subplots()
    ax.imshow(
        dl_d_logits.detach(),
        cmap="gray",
    )
    ax.set_xlabel("Class (C)")
    ax.set_ylabel("Batch (N)")
    ax.set_title("Gradient of logits")
    ax.grid(False)
    plt.show()
