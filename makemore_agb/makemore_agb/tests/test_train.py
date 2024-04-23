"""Contains tests for the train module."""

from itertools import chain

from makemore_mlp.data_classes import ModelParams, OptimizationParams
from makemore_mlp.models import get_model
from makemore_mlp.preprocessing import get_train_validation_and_test_set
from makemore_mlp.train import parse_args, train_neural_net_model


def test_train_neural_net_model() -> None:
    """Test the test_train_neural_net_model function."""
    model_params = ModelParams(block_size=3, embedding_size=2, hidden_layer_neurons=100)

    # Obtain the data
    (
        training_input,
        training_output,
        _,
        _,
        _,
        _,
    ) = get_train_validation_and_test_set(block_size=model_params.block_size)

    # Obtain the model
    model = get_model(
        block_size=model_params.block_size,
        embedding_size=model_params.embedding_size,
        hidden_layer_neurons=model_params.hidden_layer_neurons,
    )

    # Set the model options
    mini_batches_per_iteration = 1
    optimization_params = OptimizationParams(
        total_mini_batches=0,  # Not in affect here
        mini_batches_per_iteration=mini_batches_per_iteration,
        batch_size=32,
        learning_rate=lambda _: 0.1,
    )

    # Train for one step
    model, loss, step = train_neural_net_model(
        model=model,
        input_training_data=training_input,
        ground_truth_data=training_output,
        optimization_params=optimization_params,
    )

    assert len(loss) == mini_batches_per_iteration
    assert len(step) == mini_batches_per_iteration
    assert optimization_params.cur_mini_batch == 1

    # Train the model again with changed parameters
    mini_batches_per_iteration = 2
    optimization_params.batch_size = 64
    optimization_params.mini_batches_per_iteration = 2
    optimization_params.learning_rate = lambda _: 00.1

    # Train for one step
    model, loss, step = train_neural_net_model(
        model=model,
        input_training_data=training_input,
        ground_truth_data=training_output,
        optimization_params=optimization_params,
    )

    assert len(loss) == mini_batches_per_iteration
    assert len(step) == mini_batches_per_iteration
    assert optimization_params.cur_mini_batch == 3


def test_parse_args() -> None:
    """Test that the arg parsing works"""
    # Test the long arguments
    ground_truth = {
        "--block-size": 1,
        "--embedding-size": 2,
        "--hidden-layer-neurons": 3,
        "--total-mini-batches": 4,
        "--mini-batches-per-iteration": 5,
        "--batch-size": 5,
    }
    long_short_map = {
        "--block-size": "-s",
        "--embedding-size": "-e",
        "--hidden-layer-neurons": "-l",
        "--total-mini-batches": "-t",
        "--mini-batches-per-iteration": "-m",
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
        "--total-mini-batches": args.total_mini_batches,
        "--mini-batches-per-iteration": args.mini_batches_per_iteration,
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
