"""Module for models."""

import torch
from makemore_wavenet.data_classes import ModelParams
from makemore_wavenet.module import Sequential
from makemore_wavenet.ops.batchnorm1d import BatchNorm1d
from makemore_wavenet.ops.embedding import Embedding
from makemore_wavenet.ops.flatten import Flatten
from makemore_wavenet.ops.linear import Linear
from makemore_wavenet.ops.tanh import Tanh

from makemore_wavenet import VOCAB_SIZE


def get_model(
    model_params: ModelParams,
) -> Sequential:
    """Return the pytorch model.

    Args:
        model_params (ModelParams): The parameters of the model

    Raises:
        TypeError: If last layer is not Linear

    Returns:
        Sequential: The sequence which makes up the model.
    """
    # NOTE: When we are using batch normalization the biases will get
    #       cancelled out by subtracting batch_normalization_mean, and the
    #       gradient will become zero.
    #       batch_normalization_bias will in any case play the role as bias
    #       in the pre activation layers.
    model = Sequential(
        [
            Embedding(
                num_embeddings=VOCAB_SIZE, embedding_dim=model_params.embedding_size
            ),
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
    )

    # Make the last layer less confident
    with torch.no_grad():
        if not isinstance(model.layers[-1], Linear):
            raise TypeError("Last layer is expected to be linear")
        model.layers[-1].weight *= 0.1

    # Make it possible to train
    for parameter in model.parameters():
        parameter.requires_grad = True

    print(
        "Number of elements in model: "
        f"{sum(layer.nelement() for layer in model.parameters())}"
    )

    return model
