"""Module for evaluation."""

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from makemore_backprop_ninja.data_classes import BatchNormalizationParameters
from makemore_backprop_ninja.module import Module
from makemore_backprop_ninja.predict import predict_neural_network


def evaluate(
    model_type: Literal["explicit", "pytorch"],
    model: Union[Tuple[torch.Tensor, ...], Tuple[Module, ...]],
    input_data: torch.Tensor,
    ground_truth: torch.Tensor,
    batch_normalization_parameters: Optional[BatchNormalizationParameters] = None,
) -> float:
    """Evaluate the on a given data set.

    Args:
        model_type (Literal["explicit", "pytorch"]): What model type to use
        model (Union[Tuple[torch.Tensor, ...], Tuple[Module, ...]]): The model
        input_data (torch.Tensor): The data to do the prediction on
        ground_truth (torch.Tensor): The ground truth of the predictions
        batch_normalization_parameters (Optional[BatchNormalizationParameters]):
            If set: Contains the running mean and the running standard deviation

    Returns:
        float: The loss
    """
    # Note the [0] as predict always returns a tuple
    logits = predict_neural_network(
        model_type=model_type,
        model=model,
        input_data=input_data,
        batch_normalization_parameters=batch_normalization_parameters,
        training=False,
    )[0]
    loss = F.cross_entropy(logits, ground_truth)
    return loss.item()
