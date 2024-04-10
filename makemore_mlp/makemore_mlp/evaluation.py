"""Module for evaluation."""

from typing import Tuple

import torch
import torch.nn.functional as F
from makemore_mlp.inference import predict_neural_network


def evaluate(
    model: Tuple[torch.Tensor, ...],
    input_data: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """Evaluate the on a given data set.

    Args:
        model (Tuple[torch.Tensor, ...]): The model
        input_data (torch.Tensor): The data to do the prediction on
        ground_truth (torch.Tensor): The ground truth of the predictions

    Returns:
        float: The loss
    """
    logits = predict_neural_network(model=model, input_data=input_data)
    loss = F.cross_entropy(logits, ground_truth)
    return loss.item()
