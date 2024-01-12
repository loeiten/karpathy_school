"""Contains tests for the inference helper module."""

import torch
from makemore_bigram.utils.inference_helper import (
    neural_net_inference_without_interpretation,
)


def test_neural_net_inference_without_interpretation() -> None:
    """Test the neural_net_inference_without_interpretation function."""
    # Mock the data
    input_data = torch.tensor([[-0.5, 0, 0.5], [1, 0, 1]])  # [2, 3]
    weights = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
    )  # [3, 4]
    probabilities = neural_net_inference_without_interpretation(
        input_data=input_data, weights=weights
    )
    assert probabilities.shape == torch.Size([2, 4])
    for i in range(probabilities.shape[0]):
        assert torch.isclose(probabilities[i, :].sum(), torch.tensor(1.0)).item()
