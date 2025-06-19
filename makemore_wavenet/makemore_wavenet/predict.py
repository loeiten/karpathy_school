"""Module to run predict on the model."""

from typing import Tuple

import torch
from makemore_wavenet.module import Module


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-arguments
def predict_neural_network(
    model: Tuple[Module, ...],
    input_data: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Predict using the pytorch-like model.

    Args:
        model (Tuple[Module, ...]): The model (Modules) to use
        input_data (torch.Tensor): The data to run inference on.
            This data has the shape (batch_size, block_size)

    Returns:
        torch.Tensor: The achieved logits with shape (batch_size)
    """
    c = model[0]
    # NOTE: c has dimension (VOCAB_SIZE, embedding_size)
    #       input_data has the dimension (batch_size, block_size)
    #       c[input_data] will grab embedding_size vectors for each of the
    #       block_size characters
    #       The dimension of emb is therefore
    #       (batch_size, block_size, embedding_size)
    embedding = c(input_data)
    # The block needs to be concatenated before multiplying it with the
    # weight
    # That is, the dimension size will be block_size*embedding_size
    # Another way to look at it is that we need the batch size to stay the
    # same, whereas the second dimension should be the rest should be squashed
    # together
    concatenated_embedding = embedding.view(embedding.shape[0], -1)
    # Alias
    x = concatenated_embedding
    for layer in model[1:]:
        x = layer(x)
    logits = x

    return (logits,)
