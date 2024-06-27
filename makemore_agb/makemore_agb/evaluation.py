"""Module for evaluation."""

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_agb.data_classes import BatchNormalizationParameters
from makemore_agb.predict import predict_neural_network


def evaluate(
    model: Tuple[torch.Tensor, ...],
    input_data: torch.Tensor,
    ground_truth: torch.Tensor,
    batch_normalization_parameters: Optional[BatchNormalizationParameters] = None,
    model_type: Literal["explicit", "pytorch"] = "explicit",
) -> float:
    """Evaluate the on a given data set.

    Args:
        model (Tuple[torch.Tensor, ...]): The model
        input_data (torch.Tensor): The data to do the prediction on
        ground_truth (torch.Tensor): The ground truth of the predictions
        batch_normalization_parameters (Optional[BatchNormalizationParameters]):
            If set: Contains the running mean and the running standard deviation
        model_type (Literal["explicit", "pytorch"]): What model type to use

    Returns:
        float: The loss
    """
    # Note the [0] as predict always returns a tuple
    logits = predict_neural_network(
        model=model,
        input_data=input_data,
        batch_normalization_parameters=batch_normalization_parameters,
        training=False,
        model_type=model_type,
    )[0]
    loss = F.cross_entropy(logits, ground_truth)
    return loss.item()
