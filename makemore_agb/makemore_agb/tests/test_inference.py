"""Contains tests for the inference module."""

import pytest
from makemore_agb.inference import parse_args, run_inference
from makemore_agb.models import get_model


@pytest.mark.parametrize("batch_normalize", [True, False])
def test_run_inference(batch_normalize: bool) -> None:
    """
    Test the run_inference function.

    Args:
        batch_normalize (bool): Whether or not to use batch normalization
    """
    # Obtain the model with default parameters
    model = get_model(block_size=3)
    # Run inference on the untrained model
    predictions = run_inference(
        model=model, n_samples=2, batch_normalize=batch_normalize
    )
    assert len(predictions) == 2


def test_parse_args() -> None:
    """Test that the arg parsing works"""
    # Test the long arguments
    n_predictions = 2
    long_arguments = ["--n-predictions", f"{n_predictions}"]
    args = parse_args(long_arguments)
    assert args.n_predictions == n_predictions

    short_arguments = []
    for arg in long_arguments:
        if arg.startswith("--"):
            short_arguments.append(arg[1:3])
        else:
            short_arguments.append(arg)

    args = parse_args(short_arguments)
    assert args.n_predictions == n_predictions
