"""Contains tests for the train module."""

from itertools import chain

import pytest
import torch
from makemore_backprop_ninja.data_classes import (
    BatchNormalizationParameters,
    ModelParams,
    OptimizationParams,
)
from makemore_backprop_ninja.models import get_explicit_model
from makemore_backprop_ninja.preprocessing import get_dataset
from makemore_backprop_ninja.train import parse_args, train_neural_net_model

from makemore_backprop_ninja import DEVICE


@pytest.mark.parametrize("use_functional", [True, False])
def test_train_neural_net_model(use_functional: bool) -> None:
    """
    Test the test_train_neural_net_model function.

    Args:
        use_functional (bool): Whether or not to use the functional version of
            the cross entropy.
            If False, the hand-written version will be used
    """
    model_params = ModelParams(
        block_size=3,
        embedding_size=2,
        hidden_layer_neurons=100,
    )

    # Obtain the data
    dataset = get_dataset(block_size=model_params.block_size)

    # Obtain the model
    model = get_explicit_model(model_params)
    batch_normalization_parameters = BatchNormalizationParameters(
        running_mean=torch.zeros(
            (1, model_params.hidden_layer_neurons),
            requires_grad=False,
            device=DEVICE,
        ),
        running_std=torch.ones(
            (1, model_params.hidden_layer_neurons),
            requires_grad=False,
            device=DEVICE,
        ),
    )

    # Set the model options
    mini_batches_per_data_capture = 1
    n_mini_batches = 1
    optimization_params = OptimizationParams(
        n_mini_batches=n_mini_batches,
        mini_batches_per_data_capture=mini_batches_per_data_capture,
        batch_size=32,
        learning_rate=lambda _: 0.1,
    )

    # Train for one step without train_statistics
    model = train_neural_net_model(
        model=model,
        dataset=dataset,
        optimization_params=optimization_params,
        batch_normalization_parameters=batch_normalization_parameters,
        use_functional=use_functional,
    )
    assert optimization_params.cur_step == 1

    # Train the model again with changed parameters
    optimization_params.n_mini_batches = 3
    optimization_params.batch_size = 64
    optimization_params.mini_batches_per_data_capture = 2
    optimization_params.learning_rate = lambda _: 00.1

    # Train for one step
    model = train_neural_net_model(
        model=model,
        dataset=dataset,
        optimization_params=optimization_params,
        batch_normalization_parameters=batch_normalization_parameters,
        use_functional=use_functional,
    )
    assert optimization_params.cur_step == 4


def test_parse_args() -> None:
    """Test that the arg parsing works"""
    # Test the long arguments
    ground_truth = {
        "--block-size": 1,
        "--embedding-size": 2,
        "--hidden-layer-neurons": 3,
        "--n-mini-batches": 4,
        "--mini-batches-per-data-capture": 5,
        "--batch-size": 5,
    }
    long_short_map = {
        "--block-size": "-s",
        "--embedding-size": "-e",
        "--hidden-layer-neurons": "-l",
        "--n-mini-batches": "-n",
        "--mini-batches-per-data-capture": "-c",
        "--batch-size": "-b",
    }
    arguments = list(
        chain.from_iterable([(key, str(value)) for key, value in ground_truth.items()])
    )
    args = parse_args(arguments)
    parsed_map = {
        "--block-size": args.block_size,
        "--embedding-size": args.embedding_size,
        "--hidden-layer-neurons": args.hidden_layer_neurons,
        "--n-mini-batches": args.n_mini_batches,
        "--mini-batches-per-data-capture": args.mini_batches_per_data_capture,
        "--batch-size": args.batch_size,
    }

    for arg, val in ground_truth.items():
        assert val == parsed_map[arg]

    arguments = list(
        chain.from_iterable(
            [(long_short_map[key], str(value)) for key, value in ground_truth.items()]
        )
    )
    args = parse_args(arguments)
    for arg, val in ground_truth.items():
        assert val == parsed_map[arg]
