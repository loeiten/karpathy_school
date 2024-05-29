"""Contains tests for the distribution diagnostics module."""

import pytest
from makemore_agb.distribution_diagnostics import parse_args, plot_initial_distributions


@pytest.mark.parametrize("good_initialization", [True, False])
def test_plot_initial_distributions(good_initialization: bool) -> None:
    """Test the plot_initial_distributions function.

    Args:
        good_initialization (bool): Whether or not to use a good distribution
            for the initialization weights
    """
    plot_initial_distributions(good_initialization=good_initialization, show=False)


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
