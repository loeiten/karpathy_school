"""Module for evaluation."""

import torch
import torch.nn.functional as F
from gpt_from_scratch.module import Sequential


def evaluate(
    model: Sequential,
    input_data: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """Evaluate the on a given data set.

    Args:
        model (Tuple[Module, ...]): The model
        input_data (torch.Tensor): The data to do the prediction on
        ground_truth (torch.Tensor): The ground truth of the predictions

    Returns:
        float: The loss
    """
    logits = model(input_data)
    loss = F.cross_entropy(logits, ground_truth)
    return loss.item()
