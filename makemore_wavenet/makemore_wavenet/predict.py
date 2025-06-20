"""Module to run predict on the model."""

from typing import Tuple

import torch
from makemore_wavenet.module import Module


def predict_neural_network(
    model: Tuple[Module, ...],
    input_data: torch.Tensor,
) -> torch.Tensor:
    """Predict using the pytorch-like model.

    Args:
        model (Tuple[Module, ...]): The model (Modules) to use
        input_data (torch.Tensor): The data to run inference on.
            This data has the shape (batch_size, block_size)

    Returns:
        torch.Tensor: The achieved logits with shape (batch_size)
    """
    # Alias
    x = input_data
    for layer in model:
        x = layer(x)
    logits = x

    return logits
