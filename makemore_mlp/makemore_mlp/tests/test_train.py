"""Contains tests for the models module."""

from makemore_mlp.data_classes import ModelParams, OptimizationParams
from makemore_mlp.models import get_model
from makemore_mlp.preprocessing import get_train_validation_and_test_set
from makemore_mlp.train import train_neural_net_model


def test_train_neural_net_model() -> None:
    """Test the test_train_neural_net_model function."""
    model_params = ModelParams(block_size=3, embedding_size=2, hidden_layer_neurons=100)

    # Obtain the data
    (
        train_input,
        train_output,
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
    total_mini_batches = 1
    optimization_params = OptimizationParams(
        total_mini_batches=total_mini_batches, batch_size=32, learning_rate=0.1
    )

    # Train for one step
    model, loss, step = train_neural_net_model(
        model=model,
        input_training_data=train_input,
        ground_truth_data=train_output,
        optimization_params=optimization_params,
    )

    assert len(loss) == total_mini_batches
    assert len(step) == total_mini_batches

    # Train the model again with changed parameters
    total_mini_batches = 2
    optimization_params = OptimizationParams(
        total_mini_batches=total_mini_batches, batch_size=64, learning_rate=00.1
    )

    # Train for one step
    model, loss, step = train_neural_net_model(
        model=model,
        input_training_data=train_input,
        ground_truth_data=train_output,
        optimization_params=optimization_params,
    )

    assert len(loss) == total_mini_batches
    assert len(step) == total_mini_batches
