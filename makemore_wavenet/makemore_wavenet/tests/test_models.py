"""Contains tests for the models module."""

import typing

import pytest
import torch
from makemore_wavenet.data_classes import ModelParams
from makemore_wavenet.models import get_model
from makemore_wavenet.ops.tanh import Tanh

from makemore_wavenet import VOCAB_SIZE


# We disable mypy check as layers have different attributes
@typing.no_type_check
@pytest.mark.parametrize(
    "block_size, embedding_size, hidden_layer_neurons",
    ((3, 2, 100), (8, 15, 200), (20, 3, 5)),
)
@pytest.mark.skip(reason="Need to enable this once fixed")
# Need the locals
# pylint: disable-next=too-many-locals
def test_get_model(
    block_size: int, embedding_size: int, hidden_layer_neurons: int
) -> None:
    """Test the get_model function.

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

    model = get_model(model_params=model_params)

    layered_parameters = [p for layer in model for p in layer.parameters()]

    # Check the number of parameters
    assert (
        sum(parameters.nelement() for parameters in layered_parameters)
        == (VOCAB_SIZE * embedding_size)  # size of c
        + (block_size * embedding_size * hidden_layer_neurons)  # size of w1
        + (hidden_layer_neurons)  # size of b1
        + 2 * hidden_layer_neurons  # size of batchn1
        + (
            (hidden_layer_neurons * hidden_layer_neurons)  # size of wx
            + hidden_layer_neurons  # size of bx
            + 2 * hidden_layer_neurons  # size of batchnx
        )
        * 4  # Repeated 4 times
        + (hidden_layer_neurons * VOCAB_SIZE)  # size of wn
        + 0  # size of bn
        + 2 * VOCAB_SIZE  # size of batchnn
    )

    c, l1, b1, t1, l2, b2, t2, l3, b3, t3, l4, b4, t4, l5, b5, t5, l6, b6 = model

    assert c.weight.shape == torch.Size([VOCAB_SIZE, embedding_size])
    assert l1.weight.shape == torch.Size(
        [block_size * embedding_size, hidden_layer_neurons]
    )
    assert l1.bias.shape == torch.Size([hidden_layer_neurons])
    assert b1.gamma.shape == torch.Size([hidden_layer_neurons])
    assert b1.beta.shape == torch.Size([hidden_layer_neurons])
    assert isinstance(t1, Tanh)
    hidden_layers = [l2, l3, l4, l5]
    hidden_batch_normalizations = [b2, b3, b4, b5]
    hidden_activations = [t2, t3, t4, t5]
    for layer, batch_normalization, activation in zip(
        hidden_layers, hidden_batch_normalizations, hidden_activations
    ):
        assert layer.weight.shape == torch.Size(
            [hidden_layer_neurons, hidden_layer_neurons]
        )
        assert layer.bias.shape == torch.Size([hidden_layer_neurons])
        assert batch_normalization.gamma.shape == torch.Size([hidden_layer_neurons])
        assert batch_normalization.beta.shape == torch.Size([hidden_layer_neurons])
        assert isinstance(activation, Tanh)
    assert l6.weight.shape == torch.Size([hidden_layer_neurons, VOCAB_SIZE])
    assert l6.bias is None
    assert b6.gamma.shape == torch.Size([VOCAB_SIZE])
    assert b6.beta.shape == torch.Size([VOCAB_SIZE])
