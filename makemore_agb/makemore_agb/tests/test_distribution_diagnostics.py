"""Contains tests for the distribution diagnostics module."""

import pytest
from makemore_agb.distribution_diagnostics import parse_args, plot_initial_distributions


@pytest.mark.parametrize("batch_normalize", [True, False])
@pytest.mark.parametrize("good_initialization", [True, False])
def test_plot_initial_distributions(
    good_initialization: bool,
    batch_normalize: bool,
) -> None:
    """Test the plot_initial_distributions function.

    Args:
        good_initialization (bool): Whether or not to use a good distribution
            for the initialization weights
        batch_normalize (bool): Whether or not to use batch normalization
    """
    plot_initial_distributions(
        good_initialization=good_initialization,
        batch_normalize=batch_normalize,
        show=False,
        model_type="explicit",
    )
    with pytest.raises(NotImplementedError):
        plot_initial_distributions(
            good_initialization=good_initialization,
            batch_normalize=batch_normalize,
            show=False,
            model_type="pytorch",
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
