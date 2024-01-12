"""Module containing helping function for training."""

import torch


def neural_net_inference_without_interpretation(
    input_data: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Run inference on the neural net without interpretation of the data.

    NOTE: Although this is inference, it's located in train_helper to avoid
          circular imports.
          Interpreted inference is done through run_inference

    Args:
        input_data (torch.Tensor): The data to run the model on
        weights (torch.Tensor): The model
            (one hidden layer without bias, i.e. a matrix)

    Returns:
        torch.Tensor: The predictions
    """
    # Predict the log counts
    logits = input_data @ weights
    # This is equivalent to the count matrix
    counts = logits.exp()
    # NOTE: This is equivalent to softmax
    probabilities = counts / counts.sum(dim=1, keepdim=True)
    return probabilities
