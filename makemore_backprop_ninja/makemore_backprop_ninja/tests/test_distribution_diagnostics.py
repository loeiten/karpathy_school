"""Contains tests for the distribution diagnostics module."""

from typing import Literal

import pytest
from makemore_backprop_ninja.distribution_diagnostics import (
    parse_args,
    plot_initial_distributions,
)


@pytest.mark.parametrize("batch_normalize", [True, False])
@pytest.mark.parametrize("good_initialization", [True, False])
@pytest.mark.parametrize("model_type", ["explicit", "pytorch"])
def test_plot_initial_distributions(
    good_initialization: bool,
    batch_normalize: bool,
    model_type: Literal["explicit", "pytorch"],
) -> None:
    """Test the plot_initial_distributions function.

    Args:
        good_initialization (bool): Whether or not to use a good distribution
            for the initialization weights
        batch_normalize (bool): Whether or not to use batch normalization
        model_type (Literal["explicit", "pytorch"]): What model type to use
    """
    plot_initial_distributions(
        good_initialization=good_initialization,
        batch_normalize=batch_normalize,
        show=False,
        model_type=model_type,
    )


def test_parse_args() -> None:
    """Test that the arg parsing works"""
    # Test the long arguments
    ground_truth = {
        "--good-initialization": True,
        "--batch-normalize": True,
    }
    long_short_map = {
        "--good-initialization": "-g",
        "--batch-normalize": "-m",
    }
    arguments = list(ground_truth.keys())
    args = parse_args(arguments)
    parsed_map = {
        "--good-initialization": args.good_initialization,
        "--batch-normalize": args.batch_normalize,
    }

    for arg, val in ground_truth.items():
        assert val == parsed_map[arg]

    arguments = list(long_short_map.values())
    args = parse_args(arguments)
    for arg, val in ground_truth.items():
        assert val == parsed_map[arg]
