"""Contains tests for the metrics module."""

import torch
from makemore_bigram.metrics import (
    calculate_avg_nll_of_neural_net_model,
    calculate_avg_nll_of_probability_matrix_model,
)
from makemore_bigram.preprocessing import get_padded_data
from makemore_bigram.train import get_probability_matrix


def test_calculate_avg_nll_of_probability_matrix_model() -> None:
    """Test the calculate_avg_nll_of_probability_matrix_model function."""
    names = get_padded_data()
    probability_matrix = get_probability_matrix()
    avg_nll = calculate_avg_nll_of_probability_matrix_model(names, probability_matrix)
    assert avg_nll.item() < 2.455


def test_calculate_avg_nll_of_neural_net_model() -> None:
    """Test the calculate_avg_nll_of_neural_net_model function."""
    # Mock the data (in order to avoid training)
    probabilities = torch.Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ground_truth = torch.tensor([0, 1, 0])
    n_examples = 3
    avg_nll = calculate_avg_nll_of_neural_net_model(
        probabilities=probabilities, ground_truth=ground_truth, n_examples=n_examples
    )
    # The above is a perfect model, so the avg_nll should be 0
    assert torch.isclose(avg_nll, torch.tensor(0.0)).item()
