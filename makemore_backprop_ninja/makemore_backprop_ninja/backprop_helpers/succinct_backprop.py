"""Module containing the verbose_backprop."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable=too-many-locals,too-many-statements,too-many-lines
def succinct_manual_backprop(
    model: Tuple[torch.Tensor, ...],
    intermediate_variables: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    input_data: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Do the manual back propagation, and set the gradients to the parameters.

    Args:
        model (Tuple[torch.Tensor,...]): The weights of the model
        intermediate_variables (Dict[str, torch.Tensor]): The intermediate
            variables (i.e. those which are not part of model parameters).
        targets(torch.Tensor): The targets
            Needed to compute the log_prob gradients
        input_data (torch.Size): The input data for the batch

    Returns:
        A map of the gradients
    """
   # Alias for the model weights
    (
        c,
        w1,
        _,  # We are not using b1 in any calculations
        w2,
        _,  # We are not using b2 in any calculations
        batch_normalization_gain,
        _,  # We are not using batch_normalization_bias
    ) = model
    # Intermediate variables from predict
    embedding = intermediate_variables["embedding"]
    concatenated_embedding = intermediate_variables["concatenated_embedding"]

    inv_batch_normalization_std = intermediate_variables["inv_batch_normalization_std"]
    batch_normalization_raw = intermediate_variables["batch_normalization_raw"]
    h_pre_activation = intermediate_variables["h_pre_activation"]
    h = intermediate_variables["h"]
    # Intermediate variables from loss
    logits = intermediate_variables["logits"]

    batch_size = embedding.size(dim=0)
