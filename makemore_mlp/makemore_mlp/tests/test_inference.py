"""Contains tests for the inference module."""

from makemore_mlp.inference import run_inference
from makemore_mlp.models import get_model


def test_run_inference() -> None:
    """Test the run_inference function."""
    # Obtain the model with default parameters
    model = get_model(block_size=3)
    # Run inference on the untrained model
    predictions = run_inference(model=model, n_samples=2)
    assert len(predictions) == 2
