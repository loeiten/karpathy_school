"""Module for models."""

import torch
from gpt_from_scratch.data_classes import ModelParams
from gpt_from_scratch.module import Sequential
from gpt_from_scratch.ops.batchnorm1d import BatchNorm1dCustom
from gpt_from_scratch.ops.embedding import Embedding
from gpt_from_scratch.ops.flatten import FlattenConsecutive
from gpt_from_scratch.ops.linear import Linear
from gpt_from_scratch.ops.tanh import Tanh

from gpt_from_scratch import VOCAB_SIZE

from typing import Optional


def get_vanilla_model(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """Return the vanilla pytorch model.

    Args:
        model_params (ModelParams): The parameters of the model
        device (Optional[torch.device], optional): Device to use for the tensors.
            Defaults to None.

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
                num_embeddings=VOCAB_SIZE,
                embedding_dim=model_params.embedding_size,
                device=device,
            ),
            FlattenConsecutive(n_consecutive_elements=model_params.block_size),
            Linear(
                fan_in=model_params.embedding_size * model_params.block_size,
                fan_out=model_params.hidden_layer_neurons,
                bias=False,
                device=device,
            ),
            BatchNorm1dCustom(dim=model_params.hidden_layer_neurons, device=device),
            Tanh(),
            Linear(
                fan_in=model_params.hidden_layer_neurons,
                fan_out=VOCAB_SIZE,
                device=device,
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


def get_hierarchical_model(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """
    Return the hierarchical pytorch model.

    This is the same as a dilated causal convolution layer.
    The idea is that instead of squishing the vocabulary into a embedding, we
    will instead do so gradually in a tree based reduction matter.

    Args:
        model_params (ModelParams): The parameters of the model
        device (Optional[torch.device], optional): Device to use for the tensors.
            Defaults to None.

    Raises:
        TypeError: If last layer is not Linear

    Returns:
        Sequential: The sequence which makes up the model.
    """
    # These are hard-coded due to the architecture of the dilation
    model_params.block_size = 8
    dilation_degree = 2

    # NOTE: When we are using batch normalization the biases will get
    #       cancelled out by subtracting batch_normalization_mean, and the
    #       gradient will become zero.
    #       batch_normalization_bias will in any case play the role as bias
    #       in the pre activation layers.
    model = Sequential(
        [
            Embedding(
                num_embeddings=VOCAB_SIZE,
                embedding_dim=model_params.embedding_size,
                device=device,
            ),
            # First part
            FlattenConsecutive(n_consecutive_elements=dilation_degree),
            Linear(
                fan_in=model_params.embedding_size * dilation_degree,
                fan_out=model_params.hidden_layer_neurons,
                bias=False,
                device=device,
            ),
            BatchNorm1dCustom(dim=model_params.hidden_layer_neurons, device=device),
            Tanh(),
            # Second part
            FlattenConsecutive(n_consecutive_elements=dilation_degree),
            Linear(
                fan_in=model_params.hidden_layer_neurons * dilation_degree,
                fan_out=model_params.hidden_layer_neurons,
                bias=False,
                device=device,
            ),
            BatchNorm1dCustom(dim=model_params.hidden_layer_neurons, device=device),
            Tanh(),
            # Third part
            FlattenConsecutive(n_consecutive_elements=dilation_degree),
            Linear(
                fan_in=model_params.hidden_layer_neurons * dilation_degree,
                fan_out=model_params.hidden_layer_neurons,
                bias=False,
                device=device,
            ),
            BatchNorm1dCustom(dim=model_params.hidden_layer_neurons, device=device),
            Tanh(),
            Linear(
                fan_in=model_params.hidden_layer_neurons,
                fan_out=VOCAB_SIZE,
                device=device,
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


def get_original_12k(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """Return the original 12k model.

    Approximate train loss: 2.058
    Approximate val loss: 2.105

    Args:
        model_params (ModelParams): The parameters of the model
            Note that these will be overwritten
        device (Optional[torch.device], optional): Device to use for the tensors.
            Defaults to None.

    Returns:
        Sequential: The sequence which makes up the model.
    """
    model_params.embedding_size = 10
    model_params.hidden_layer_neurons = 200
    model_params.block_size = 3

    return get_vanilla_model(model_params=model_params, device=device)


def get_context_8_22k(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """Return the 22k model with context length of 8.

    Approximate train loss: 1.918
    Approximate val loss: 2.033

    Args:
        model_params (ModelParams): The parameters of the model
            Note that these will be overwritten
        device (Optional[torch.device], optional): Device to use for the tensors.
            Defaults to None.

    Returns:
        Sequential: The sequence which makes up the model.
    """
    model_params.embedding_size = 10
    model_params.hidden_layer_neurons = 200
    model_params.block_size = 8

    return get_vanilla_model(model_params=model_params, device=device)


def get_hierarchical_22k(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """Return the 22k model with hierarchical embedding of the layers.

    Approximate train loss: 1.911
    Approximate val loss: 2.019

    Args:
        model_params (ModelParams): The parameters of the model
            Note that these will be overwritten

    Returns:
        Sequential: The sequence which makes up the model.
    """
    model_params.embedding_size = 10
    model_params.hidden_layer_neurons = 68

    return get_hierarchical_model(model_params=model_params, device=device)


def get_scaled_up_76k(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """Return the 76k model with hierarchical embedding of the layers.

    Approximate train loss: 1.769
    Approximate val loss: 1.993

    Args:
        model_params (ModelParams): The parameters of the model
            Note that these will be overwritten
        device (Optional[torch.device], optional): Device to use for the tensors.
            Defaults to None.

    Returns:
        Sequential: The sequence which makes up the model.
    """
    model_params.embedding_size = 24
    model_params.hidden_layer_neurons = 128

    return get_hierarchical_model(model_params=model_params, device=device)


def get_scaled_up_77k(
    model_params: ModelParams,
    device: Optional[torch.device] = None,
) -> Sequential:
    """Return the 77k model with hierarchical embedding of the layers.

    Approximate train loss: 1.766
    Approximate val loss: 1.984

    Args:
        model_params (ModelParams): The parameters of the model
            Note that these will be overwritten
        device (Optional[torch.device], optional): Device to use for the tensors.
            Defaults to None.

    Returns:
        Sequential: The sequence which makes up the model.
    """
    model_params.embedding_size = 26
    model_params.hidden_layer_neurons = 128

    return get_hierarchical_model(model_params=model_params, device=device)
