"""Contains tests for the distribution diagnostics module."""

import pytest
from makemore_agb.distribution_diagnostics import plot_initial_distributions


@pytest.mark.parametrize("good_initialization", [True, False])
def test_plot_initial_distributions(good_initialization: bool) -> None:
    """Test the plot_initial_distributions function.

    Args:
        good_initialization (bool): Whether or not to use a good distribution
            for the initialization weights
    """
    plot_initial_distributions(good_initialization=good_initialization, show=False)
