"""Module for models."""

from typing import Tuple

import torch
from makemore_backprop_ninja.data_classes import ModelParams

from makemore_backprop_ninja import DEVICE, VOCAB_SIZE


def get_explicit_model(model_params: ModelParams) -> Tuple[torch.Tensor, ...]:
    """Return the explicit model.

    Args:
        model_params (ModelParams): The parameters of the model

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the parameters of the
            neural net.
    """
    g = torch.Generator(device=DEVICE).manual_seed(model_params.seed)

    # NOTE: randn draws from normal distribution, whereas rand draws from a
    #       uniform distribution
    c = torch.randn(
        (VOCAB_SIZE, model_params.embedding_size),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    w1 = torch.randn(
        (
            model_params.block_size * model_params.embedding_size,
            model_params.hidden_layer_neurons,
        ),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    b1 = torch.randn(
        model_params.hidden_layer_neurons,
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    w2 = torch.randn(
        (model_params.hidden_layer_neurons, VOCAB_SIZE),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    b2 = torch.randn(VOCAB_SIZE, generator=g, requires_grad=True, device=DEVICE)

    # Initially the model is confidently wrong, that is: The probability
    # distribution of the output is not uniform
    # Recall that the logits are given as h @ w2 + b2
    # Taking the softmax of the logits gives the probability
    # The negative of the logarithm of the softmax gives the loss
    # If the model has a high value for the logit of the correct character
    # compared to all the others this will give a low loss
    # If on the other hand it assigns a high probability to the wrong
    # character, this will result in a high loss
    # If we have initialized with uniform probability, the probability of
    # drawing a specific character would be 1/VOCAB_SIZE
    # If vocab size is 27, this means that -log(1/27)=3.2958 is the log that
    # we would expect.
    # However, we usually observe a much higher loss initially as the
    # initialization by chance are favouring some characters
    # Initially we would like the logits to be close to zero due to
    # numerical stability
    b2.data *= 0
    # As we want the logits to be small, we could have set the initial
    # weights to 0
    # However, we want to break the symmetry for better training, so it's
    # better to have some entropy with numbers close to 0
    # Since we want the weights to be well behaved in both the forward and
    # the backward propagation, we set the initialization to the Kaiming
    # initialization
    w2.data *= (5 / 3) / (model_params.hidden_layer_neurons**0.5)
    # We could have some small entropy in the weights as well
    b1.data *= 0.01
    # In the pre-activation we are multiplying the embedding with some
    # random weights w1
    # This causes the product to broaden the distribution
    # Since the distribution is broad, a lot of values are in the extremes
    # of tanh
    # Looking at the gradient, we see that
    # grad = (1 - tanh(h)**2) * out.grad
    # If tanh(h) becomes -1 or 1, then the gradient becomes 0, killing all
    # possibilities to learn
    # Hence, we need to squash the distribution so that we don't hit the
    # extremes of tanh
    # It would be nice to have the standard deviation of the pre-activation
    # would be around 1 so that tanh in the activation doesn't take on
    # extreme values
    # Kaiming initialization scales the distribution so that the standard
    # deviation are well behaved both in the forward and the backward
    # propagation
    w1.data *= (5 / 3) / (
        (model_params.block_size * model_params.embedding_size) ** 0.5
    )

    parameters = [c, w1, b1, w2, b2]

    # We would like to normalize each batch after each layer so that it's
    # roughly normal
    # However, only having normal distribution would yield poor results
    # Hence we let the gain and bias be trainable parameters the network can use
    # in order to move the distribution around
    batch_normalization_gain = torch.ones(
        (1, model_params.hidden_layer_neurons), device=DEVICE
    )
    batch_normalization_bias = torch.zeros(
        (1, model_params.hidden_layer_neurons), device=DEVICE
    )

    parameters.append(batch_normalization_gain)
    parameters.append(batch_normalization_bias)

    # Make it possible to train
    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return tuple(parameters)
