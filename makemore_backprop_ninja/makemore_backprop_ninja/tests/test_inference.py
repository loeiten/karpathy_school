"""Contains tests for the inference module."""

import torch
from makemore_backprop_ninja.data_classes import (
    BatchNormalizationParameters,
    ModelParams,
)
from makemore_backprop_ninja.inference import parse_args, run_inference
from makemore_backprop_ninja.models import get_explicit_model

from makemore_backprop_ninja import DEVICE


def test_run_inference() -> None:
    """Test the run_inference function."""
    # Obtain the model with default parameters
    hidden_layer_neurons = 100

    model_params = ModelParams(
        block_size=3,
        hidden_layer_neurons=hidden_layer_neurons,
    )
    model = get_explicit_model(model_params)
    batch_normalization_parameters = BatchNormalizationParameters(
        running_mean=torch.zeros(
            (1, hidden_layer_neurons),
            requires_grad=False,
            device=DEVICE,
        ),
        running_std=torch.ones(
            (1, hidden_layer_neurons),
            requires_grad=False,
            device=DEVICE,
        ),
    )
    # Run inference on the untrained model
    predictions = run_inference(
        model=model,
        n_samples=2,
        batch_normalization_parameters=batch_normalization_parameters,
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
