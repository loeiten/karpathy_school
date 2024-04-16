"""Contains tests for the train module."""

from makemore_mlp.data_classes import ModelParams, OptimizationParams
from makemore_mlp.models import get_model
from makemore_mlp.preprocessing import get_train_validation_and_test_set
from makemore_mlp.train import train_neural_net_model


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
