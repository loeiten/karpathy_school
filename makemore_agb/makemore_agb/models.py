"""Module for models."""

from typing import Tuple

import torch
from makemore_agb.batchnorm1d import BatchNorm1d
from makemore_agb.linear import Linear
from makemore_agb.tanh import Tanh

from makemore_agb import DEVICE, VOCAB_SIZE


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-locals,too-many-arguments
def get_explicit_model(
    block_size: int,
    embedding_size: int = 2,
    hidden_layer_neurons: int = 100,
    seed: int = 2147483647,
    good_initialization: bool = True,
    batch_normalize: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Return the explicit model.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length
        embedding_size (int): The size of the embedding
        hidden_layer_neurons (int): The seed for the random number generator
        seed (int): The seed for the random number generator
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights
        batch_normalize (bool): Whether or not to include batch normalization
            parameters

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the parameters of the
            neural net.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    # NOTE: randn draws from normal distribution, whereas rand draws from a
    #       uniform distribution
    c = torch.randn(
        (VOCAB_SIZE, embedding_size), generator=g, requires_grad=True, device=DEVICE
    )
    w1 = torch.randn(
        (block_size * embedding_size, hidden_layer_neurons),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    b1 = torch.randn(
        hidden_layer_neurons, generator=g, requires_grad=True, device=DEVICE
    )
    w2 = torch.randn(
        (hidden_layer_neurons, VOCAB_SIZE),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    b2 = torch.randn(VOCAB_SIZE, generator=g, requires_grad=True, device=DEVICE)

    if good_initialization:
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
        w2.data *= (5 / 3) / (hidden_layer_neurons**0.5)
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
        w1.data *= (5 / 3) / ((block_size * embedding_size) ** 0.5)

    parameters = [c, w1, b1, w2, b2]

    if batch_normalize:
        # We would like to normalize each batch after each layer so that it's
        # roughly normal
        # However, only having normal distribution would yield poor results
        # Hence we let the gain and bias be trainable parameters the network can use
        # in order to move the distribution around
        batch_normalization_gain = torch.ones((1, hidden_layer_neurons), device=DEVICE)
        batch_normalization_bias = torch.zeros((1, hidden_layer_neurons), device=DEVICE)

        parameters.append(batch_normalization_gain)
        parameters.append(batch_normalization_bias)

    # Make it possible to train
    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return tuple(parameters)


def get_pytorch_model(
    block_size: int,
    embedding_size: int = 2,
    hidden_layer_neurons: int = 100,
    seed: int = 2147483647,
    good_initialization: bool = True,
    batch_normalize: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Return the pytorch model.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length
        embedding_size (int): The size of the embedding
        hidden_layer_neurons (int): The seed for the random number generator
        seed (int): The seed for the random number generator
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights
        batch_normalize (bool): Whether or not to include batch normalization
            parameters

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the parameters of the
            neural net.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = torch.randn(
        (VOCAB_SIZE, embedding_size), generator=g, requires_grad=True, device=DEVICE
    )

    if batch_normalize:
        # NOTE: When we are using batch normalization the biases will get
        #       cancelled out by subtracting batch_normalization_mean, and the
        #       gradient will become zero.
        #       batch_normalization_bias will in any case play the role as bias
        #       in the pre activation layers.
        layers = [
            Linear(
                fan_in=embedding_size * block_size,
                fan_out=hidden_layer_neurons,
                bias=False,
            ),
            BatchNorm1d(dim=hidden_layer_neurons),
            Tanh(),
            Linear(
                fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons, bias=False
            ),
            BatchNorm1d(dim=hidden_layer_neurons),
            Tanh(),
            Linear(
                fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons, bias=False
            ),
            BatchNorm1d(dim=hidden_layer_neurons),
            Tanh(),
            Linear(
                fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons, bias=False
            ),
            BatchNorm1d(dim=hidden_layer_neurons),
            Tanh(),
            Linear(
                fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons, bias=False
            ),
            BatchNorm1d(dim=hidden_layer_neurons),
            Tanh(),
            Linear(fan_in=hidden_layer_neurons, fan_out=VOCAB_SIZE, bias=False),
            BatchNorm1d(VOCAB_SIZE),
        ]
    else:
        layers = [
            Linear(fan_in=embedding_size * block_size, fan_out=hidden_layer_neurons),
            Tanh(),
            Linear(fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons),
            Tanh(),
            Linear(fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons),
            Tanh(),
            Linear(fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons),
            Tanh(),
            Linear(fan_in=hidden_layer_neurons, fan_out=hidden_layer_neurons),
            Tanh(),
            Linear(fan_in=hidden_layer_neurons, fan_out=VOCAB_SIZE),
        ]

    if good_initialization:
        # FIXME: You are here
        pass

    parameters = [c] + [
        params for layer in layers for params in layer.parameters()  # type: ignore
    ]

    # Make it possible to train
    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return tuple(parameters)
