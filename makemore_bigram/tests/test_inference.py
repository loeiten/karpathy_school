"""Contains tests for the inference module."""

import pytest
from makemore_bigram.inference import run_inference
from makemore_bigram.train import get_neural_net, get_probability_matrix

from makemore_bigram import ModelTypes


@pytest.mark.parametrize("model_type", ModelTypes.__args__)  # type: ignore
def test_run_inference(model_type: ModelTypes) -> None:
    """Test the run_inference function.

    Args:
        model_type (ModelTypes): The model type to run inference on
    """
    if model_type == "probability_matrix":
        matrix = get_probability_matrix()
    elif model_type == "neural_net":
        matrix = get_neural_net(epochs=1)

    n_samples = 5
    samples = run_inference(model_type=model_type, matrix=matrix, n_samples=n_samples)
    assert len(samples) == n_samples
