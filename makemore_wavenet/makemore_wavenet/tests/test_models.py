"""Contains tests for the models module."""

import typing

import pytest
import torch
from makemore_wavenet.data_classes import ModelParams
from makemore_wavenet.models import get_vanilla_model
from makemore_wavenet.ops.tanh import Tanh

from makemore_wavenet import VOCAB_SIZE


# We disable mypy check as layers have different attributes
@typing.no_type_check
@pytest.mark.parametrize(
    "block_size, embedding_size, hidden_layer_neurons",
    ((3, 2, 100), (8, 15, 200), (20, 3, 5)),
)
# Need the locals
def test_get_vanilla_model(
    block_size: int, embedding_size: int, hidden_layer_neurons: int
) -> None:
    """Test the get_vanilla_model function.

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

    model = get_vanilla_model(model_params=model_params)

    # Check the number of parameters
    assert (
        sum(parameters.nelement() for parameters in model.parameters())
        == (VOCAB_SIZE * embedding_size)  # size of c
        + (block_size * embedding_size * hidden_layer_neurons)  # size of w1
        + 2 * (hidden_layer_neurons)  # size of b1 (gamma and beta)
        + (hidden_layer_neurons * VOCAB_SIZE)  # size of w2
        + VOCAB_SIZE  # size of b2
    )

    c, _, l1, b1, t1, l2 = model.layers

    assert c.weight.shape == torch.Size([VOCAB_SIZE, embedding_size])
    assert l1.weight.shape == torch.Size(
        [block_size * embedding_size, hidden_layer_neurons]
    )
    assert l1.bias is None
    assert b1.gamma.shape == torch.Size([hidden_layer_neurons])
    assert b1.beta.shape == torch.Size([hidden_layer_neurons])
    assert isinstance(t1, Tanh)
    assert l2.weight.shape == torch.Size([hidden_layer_neurons, VOCAB_SIZE])
    assert l2.bias.shape == torch.Size([VOCAB_SIZE])
