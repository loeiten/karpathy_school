"""Contains tests for the models module."""

import pytest
import torch
from makemore_backprop_ninja.data_classes import ModelParams
from makemore_backprop_ninja.models import get_explicit_model

from makemore_backprop_ninja import VOCAB_SIZE


@pytest.mark.parametrize(
    "block_size, embedding_size, hidden_layer_neurons",
    ((3, 2, 100), (8, 15, 200), (20, 3, 5)),
)
def test_get_explicit_model(
    block_size: int, embedding_size: int, hidden_layer_neurons: int
) -> None:
    """Test the get_explicit_model function.

    Args:
        block_size (int): The context length
        embedding_size (int): Size of the embedding space
        hidden_layer_neurons (int): Number of neurons in the hidden layer
    """
    model_params = ModelParams(
        block_size=block_size,
        embedding_size=embedding_size,
        hidden_layer_neurons=hidden_layer_neurons,
    )

    model = get_explicit_model(model_params=model_params)

    # Check the number of parameters
    assert (
        sum(parameters.nelement() for parameters in model)
        == (VOCAB_SIZE * embedding_size)  # size of c
        + (block_size * embedding_size * hidden_layer_neurons)  # size of w1
        + (hidden_layer_neurons)  # size of b1
        + (hidden_layer_neurons * VOCAB_SIZE)  # size of w2
        + VOCAB_SIZE  # size of b2
        + hidden_layer_neurons  # size of batch_normalization_gain
        + hidden_layer_neurons  # size of batch_normalization_bias
    )

    # We know the exact dimension
    # pylint: disable-next=unbalanced-tuple-unpacking
    c, w1, b1, w2, b2, batch_normalization_gain, batch_normalization_bias = model

    assert c.shape == torch.Size([VOCAB_SIZE, embedding_size])
    assert w1.shape == torch.Size([block_size * embedding_size, hidden_layer_neurons])
    assert b1.shape == torch.Size([hidden_layer_neurons])
    assert w2.shape == torch.Size([hidden_layer_neurons, VOCAB_SIZE])
    assert b2.shape == torch.Size([VOCAB_SIZE])
    assert batch_normalization_gain.shape == torch.Size([1, hidden_layer_neurons])
    assert batch_normalization_bias.shape == torch.Size([1, hidden_layer_neurons])
