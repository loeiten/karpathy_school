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
    return {}
