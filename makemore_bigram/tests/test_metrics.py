"""Contains tests for the metrics module."""

from makemore_bigram.metrics import calculate_avg_nll_of_matrix_model
from makemore_bigram.preprocessing import get_padded_data
from makemore_bigram.train import get_probability_matrix


def test_calculate_avg_nll_of_matrix_model():
    """Test the calculate_avg_nll_of_matrix_model function."""
    names = get_padded_data()
    probability_matrix = get_probability_matrix()
    avg_nll = calculate_avg_nll_of_matrix_model(names, probability_matrix)
    assert avg_nll.item() < 2.455
