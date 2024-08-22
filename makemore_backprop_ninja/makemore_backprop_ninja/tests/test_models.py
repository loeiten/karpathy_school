"""Contains tests for the models module."""

import typing

import pytest
import torch
from makemore_backprop_ninja.data_classes import ModelParams
from makemore_backprop_ninja.models import (
    get_explicit_model,
    get_model_function,
    get_pytorch_model,
)
from makemore_backprop_ninja.tanh import Tanh

from makemore_agb import VOCAB_SIZE


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
        batch_normalize=False,
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
    )

    # We know the exact output from the batch_normalize flag
    # pylint: disable-next=unbalanced-tuple-unpacking
    c, w1, b1, w2, b2 = model

    assert c.shape == torch.Size([VOCAB_SIZE, embedding_size])
    assert w1.shape == torch.Size([block_size * embedding_size, hidden_layer_neurons])
    assert b1.shape == torch.Size([hidden_layer_neurons])
    assert w2.shape == torch.Size([hidden_layer_neurons, VOCAB_SIZE])
    assert b2.shape == torch.Size([VOCAB_SIZE])

    model_params.batch_normalize = True
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

    # We know the exact output from the batch_normalize flag
    # pylint: disable-next=unbalanced-tuple-unpacking
    c, w1, b1, w2, b2, batch_normalization_gain, batch_normalization_bias = model

    assert c.shape == torch.Size([VOCAB_SIZE, embedding_size])
    assert w1.shape == torch.Size([block_size * embedding_size, hidden_layer_neurons])
    assert b1.shape == torch.Size([hidden_layer_neurons])
    assert w2.shape == torch.Size([hidden_layer_neurons, VOCAB_SIZE])
    assert b2.shape == torch.Size([VOCAB_SIZE])
    assert batch_normalization_gain.shape == torch.Size([1, hidden_layer_neurons])
    assert batch_normalization_bias.shape == torch.Size([1, hidden_layer_neurons])


# We disable mypy check as layers have different attributes
@typing.no_type_check
@pytest.mark.parametrize(
    "block_size, embedding_size, hidden_layer_neurons",
    ((3, 2, 100), (8, 15, 200), (20, 3, 5)),
)
# Need the locals
# pylint: disable-next=too-many-locals
def test_get_pytorch_model(
    block_size: int, embedding_size: int, hidden_layer_neurons: int
) -> None:
    """Test the get_pytorch_model function.

    Args:
        block_size (int): The context length
        embedding_size (int): Size of the embedding space
        hidden_layer_neurons (int): Number of neurons in the hidden layer
    """
    model_params = ModelParams(
        block_size=block_size,
        embedding_size=embedding_size,
        hidden_layer_neurons=hidden_layer_neurons,
        batch_normalize=False,
    )

    model = get_pytorch_model(model_params=model_params)

    layered_parameters = [p for layer in model for p in layer.parameters()]

    # Check the number of parameters
    assert (
        sum(parameters.nelement() for parameters in layered_parameters)
        == (VOCAB_SIZE * embedding_size)  # size of c
        + (block_size * embedding_size * hidden_layer_neurons)  # size of w1
        + (hidden_layer_neurons)  # size of b1
        + (
            (hidden_layer_neurons * hidden_layer_neurons)  # size of wx
            + hidden_layer_neurons  # size of bx
        )
        * 4  # Repeated 4 times
        + (hidden_layer_neurons * VOCAB_SIZE)  # size of wn
        + VOCAB_SIZE  # size of bn
    )

    # We know how much we're unpacking
    # pylint: disable-next=unbalanced-tuple-unpacking
    c, l1, t1, l2, t2, l3, t3, l4, t4, l5, t5, l6 = model

    assert c.weight.shape == torch.Size([VOCAB_SIZE, embedding_size])
    assert l1.weight.shape == torch.Size(
        [block_size * embedding_size, hidden_layer_neurons]
    )
    assert l1.bias.shape == torch.Size([hidden_layer_neurons])
    assert isinstance(t1, Tanh)
    hidden_layers = [l2, l3, l4, l5]
    hidden_activations = [t2, t3, t4, t5]
    for layer, activation in zip(hidden_layers, hidden_activations):
        assert layer.weight.shape == torch.Size(
            [hidden_layer_neurons, hidden_layer_neurons]
        )
        assert layer.bias.shape == torch.Size([hidden_layer_neurons])
        assert isinstance(activation, Tanh)
    # This should be a linear layer
    # pylint: disable-next=no-member
    assert l6.weight.shape == torch.Size([hidden_layer_neurons, VOCAB_SIZE])
    # pylint: disable-next=no-member
    assert l6.bias.shape == torch.Size([VOCAB_SIZE])

    model_params.batch_normalize = True
    model = get_pytorch_model(model_params=model_params)

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


def test_get_model_function() -> None:
    """Test the get model function."""
    model_function = get_model_function("explicit")
    assert model_function.__name__ == "get_explicit_model"
    model_function = get_model_function("pytorch")
    assert model_function.__name__ == "get_pytorch_model"

    with pytest.raises(ValueError):
        get_model_function("I do not exists")  # type: ignore
