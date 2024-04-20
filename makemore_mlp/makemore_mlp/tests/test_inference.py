"""Contains tests for the inference module."""

from makemore_mlp.inference import parse_args, run_inference
from makemore_mlp.models import get_model


def test_run_inference() -> None:
    """Test the run_inference function."""
    # Obtain the model with default parameters
    model = get_model(block_size=3)
    # Run inference on the untrained model
    predictions = run_inference(model=model, n_samples=2)
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
