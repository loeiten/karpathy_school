"""Module for visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from makemore_bigram import INVERSE_ALPHABET_DICT


def create_heatmap(count_matrix: torch.Tensor, save_dir: Path) -> None:
    """Make a heatmap of the count matrix and store it.

    Args:
        count_matrix (torch.Tensor): Matrix containing frequency of preceding
            characters
        save_dir (Path): Path to save the plot to
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 16)
    # Plot the matrix
    ax.imshow(count_matrix, cmap="Blues")

    # Annotate the cells
    n_tokens = len(INVERSE_ALPHABET_DICT)
    # NOTE: x and y are here referring to the x and y of the plot
    #       For our matrix y is the row
    for x in range(n_tokens):
        for y in range(n_tokens):
            character_string = INVERSE_ALPHABET_DICT[y] + INVERSE_ALPHABET_DICT[x]
            ax.text(x, y, character_string, ha="center", va="bottom", color="gray")
            ax.text(
                x, y, count_matrix[y, x].item(), ha="center", va="top", color="gray"
            )
    plt.axis("off")

    save_path = save_dir.joinpath("heatmap.png")
    fig.savefig(save_path)
    print(f"Saved image to {save_path}")


if __name__ == "__main__":
    from makemore_bigram.preprocessing import get_count_matrix
    from makemore_bigram.utils.paths import get_output_dir

    count_matrix_ = get_count_matrix()
    output_dir_ = get_output_dir()
    create_heatmap(count_matrix=count_matrix_, save_dir=output_dir_)
