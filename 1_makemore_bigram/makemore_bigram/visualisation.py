"""Module for visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from makemore_bigram.utils.train_helper import get_count_matrix

from makemore_bigram import INDEX_TO_TOKEN


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
    n_tokens = len(INDEX_TO_TOKEN)
    # NOTE: x and y are here referring to the x and y of the plot
    #       For our matrix y is the row
    for x in range(n_tokens):
        for y in range(n_tokens):
            character_string = INDEX_TO_TOKEN[y] + INDEX_TO_TOKEN[x]
            ax.text(x, y, character_string, ha="center", va="bottom", color="gray")
            ax.text(
                x, y, count_matrix[y, x].item(), ha="center", va="top", color="gray"
            )
    plt.axis("off")

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir.joinpath("heatmap.png")
    fig.savefig(save_path)
    print(f"Saved image to {save_path}")


def main(save_dir: Path):
    """
    Run the main function of visualisation module.

    Args:
        save_dir (Path): Directory to store heatmap in
    """
    count_matrix_ = get_count_matrix()
    create_heatmap(count_matrix=count_matrix_, save_dir=save_dir)


if __name__ == "__main__":
    import argparse

    from makemore_bigram.utils.paths import get_output_dir

    output_dir_ = get_output_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save_dir",
        type=Path,
        help="Directory to store heatmap plot in",
        default=output_dir_,
    )
    args = parser.parse_args()
    main(args.save_dir)
