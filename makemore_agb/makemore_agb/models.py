"""Module for models."""

from typing import Tuple

import torch

from makemore_agb import VOCAB_SIZE


def get_model(
    block_size: int,
    embedding_size: int = 2,
    hidden_layer_neurons: int = 100,
    seed: int = 2147483647,
    good_initialization: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Return the model.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length
        embedding_size (int): The size of the embedding
        hidden_layer_neurons (int): The seed for the random number generator
        seed (int): The seed for the random number generator
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the parameters of the
            neural net.
    """
    g = torch.Generator().manual_seed(seed)

    # NOTE: randn draws from normal distribution, whereas rand draws from a
    #       uniform distribution
    c = torch.randn((VOCAB_SIZE, embedding_size), generator=g, requires_grad=True)
    w1 = torch.randn(
        (block_size * embedding_size, hidden_layer_neurons),
        generator=g,
        requires_grad=True,
    )
    b1 = torch.randn(hidden_layer_neurons, generator=g, requires_grad=True)
    w2 = torch.randn(
        (hidden_layer_neurons, VOCAB_SIZE), generator=g, requires_grad=True
    )
    b2 = torch.randn(VOCAB_SIZE, generator=g, requires_grad=True)

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
        b2 *= 0

    parameters = (c, w1, b1, w2, b2)

    # Make it possible to train
    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return parameters
