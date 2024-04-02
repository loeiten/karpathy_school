"""Contains tests for the models module."""

from makemore_mlp.models import get_model
from makemore_mlp.preprocessing import get_train_validation_and_test_set
from makemore_mlp.train import train_neural_net_model

from makemore_mlp.makemore_mlp.data_classes import ModelOptions


def test_train_neural_net_model() -> None:
    """Test the test_train_neural_net_model function."""
    block_size = 3

    # Obtain the data
    (
        train_input,
        train_output,
        _,
        _,
        _,
        _,
    ) = get_train_validation_and_test_set(block_size=block_size)

    # Obtain the model
    embedding_size = 2
    hidden_layer_neurons = 100
    model = get_model(
        block_size=block_size,
        embedding_size=embedding_size,
        hidden_layer_neurons=hidden_layer_neurons,
    )

    # Set the model options
    n_mini_batches = 1
    model_options = ModelOptions(
        n_mini_batches=n_mini_batches, batch_size=32, learning_rate=0.1
    )

    # Train for one step
    model, loss_log10, step = train_neural_net_model(
        model=model,
        input_training_data=train_input,
        ground_truth_data=train_output,
        model_options=model_options,
    )

    assert len(loss_log10) == n_mini_batches
    assert len(step) == n_mini_batches

    # Train the model again with changed parameters
    n_mini_batches = 2
    model_options = ModelOptions(
        n_mini_batches=n_mini_batches, batch_size=64, learning_rate=00.1
    )

    # Train for one step
    model, loss_log10, step = train_neural_net_model(
        model=model,
        input_training_data=train_input,
        ground_truth_data=train_output,
        model_options=model_options,
    )

    assert len(loss_log10) == n_mini_batches
    assert len(step) == n_mini_batches
