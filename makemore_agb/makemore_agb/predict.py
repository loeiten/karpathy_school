"""Module to run inference on the model."""

from typing import Tuple

import torch


def predict_neural_network(
    model: Tuple[torch.Tensor, ...],
    input_data: torch.Tensor,
    inspect_pre_activation_and_h: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Predict the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_data (torch.Tensor): The data to run inference on.
            This data has the shape (batch_size, block_size)
        inspect_pre_activation_and_h (bool): Whether or not to output the
            pre-activation and activation

    Returns:
        torch.Tensor: The achieved logits with shape (batch_size)
    """
    # Alias
    c, w1, b1, w2, b2 = model
    # NOTE: c has dimension (VOCAB_SIZE, embedding_size)
    #       input_data has the dimension (batch_size, block_size)
    #       c[input_data] will grab embedding_size vectors for each of the
    #       block_size characters
    #       The dimension of emb is therefore
    #       (batch_size, block_size, embedding_size)
    embedding = c[input_data]
    # The block needs to be concatenated before multiplying it with the
    # weight
    # That is, the dimension size will be block_size*embedding_size
    # Another way to look at it is that we need the batch size to stay the
    # same, whereas the second dimension should be the rest should be squashed
    # together
    concatenated_embedding = embedding.view(embedding.shape[0], -1)
    # NOTE: + b1 is broadcasting on the correct dimension
    h_pre_activation = (concatenated_embedding @ w1) + b1
    h = torch.tanh(h_pre_activation)
    # The logits will have dimension (batch_size, VOCAB_SIZE)
    logits = h @ w2 + b2

    if not inspect_pre_activation_and_h:
        return (logits,)
    return (logits, h_pre_activation, h)
