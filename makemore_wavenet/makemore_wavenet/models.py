"""Module for models."""

from typing import Tuple

import torch
from makemore_wavenet.data_classes import ModelParams
from makemore_wavenet.module import Module
from makemore_wavenet.ops.batchnorm1d import BatchNorm1d
from makemore_wavenet.ops.embedding import Embedding
from makemore_wavenet.ops.flatten import Flatten
from makemore_wavenet.ops.linear import Linear
from makemore_wavenet.ops.tanh import Tanh

from makemore_wavenet import VOCAB_SIZE


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-arguments
def get_model(
    model_params: ModelParams,
) -> Tuple[Module, ...]:
    """Return the pytorch model.

    Args:
        model_params (ModelParams): The parameters of the model

    Raises:
        TypeError: If last layer is not Linear

    Returns:
        Tuple[Module, ...]: A tuple containing the parameters of the
            neural net.
    """
    # NOTE: When we are using batch normalization the biases will get
    #       cancelled out by subtracting batch_normalization_mean, and the
    #       gradient will become zero.
    #       batch_normalization_bias will in any case play the role as bias
    #       in the pre activation layers.
    layers = [
        Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=model_params.embedding_size),
        Flatten(),
        Linear(
            fan_in=model_params.embedding_size * model_params.block_size,
            fan_out=model_params.hidden_layer_neurons,
            bias=False,
        ),
        BatchNorm1d(dim=model_params.hidden_layer_neurons),
        Tanh(),
        Linear(
            fan_in=model_params.hidden_layer_neurons,
            fan_out=VOCAB_SIZE,
        ),
    ]

    parameters = [
        params for layer in layers for params in layer.parameters()  # type: ignore
    ]

    # Make the last layer less confident
    with torch.no_grad():
        if not isinstance(layers[-1], Linear):
            raise TypeError("Last layer is expected to be linear")
        layers[-1].weight *= 0.1

    # Make it possible to train
    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return tuple(layers)
