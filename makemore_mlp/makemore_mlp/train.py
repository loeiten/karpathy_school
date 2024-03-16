"""Module to train the model."""

from typing import Tuple

import torch
from makemore_mlp.inference import predict_neural_network


def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    input_training_data: torch.Tensor,
    ground_truth_data: torch.Tensor,
    n_mini_batches: int,
    batch_size: int,
) -> Tuple[torch.Tensor]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_training_data (torch.Tensor): The input training data with shape
            (n_samples, block_size).
            This is the data that will be fed into the features
        ground_truth_data (torch.Tensor): The correct prediction for the input
            (the correct labels)
        n_mini_batches (int): Number of mini batches to use
        batch_size (int): The batch size to use

    Returns:
        torch.Tensor: The trained model
    """
    for _ in range(n_mini_batches):
        # Mini batch constructor
        n_samples = input_training_data.shape[0]
        idxs = torch.randint(low=0, high=n_samples, size=batch_size)

        # NOTE: input_training_data has dimension (n_samples, block_size)
        #       input_training_data[idxs] selects batch_size samples from the
        #       training data
        #       The size of input_training_data[idxs] is therefore
        #       (batch_size, block_size)
        input_data = input_training_data[idxs]
        prob = predict_neural_network(model=model, input_data=input_data)

        # Negative loss likelihood:
        # - For each of the example in the batch: torch.arange(batch_size)
        #   - Select the probability of getting the ground truth:
        #     prob[torch.arange(batch_size), ground_truth_data]
        #     The shape of this will be the batch_size
        #     In a model that perfectly predicts the characters all entries in
        #     prob[torch.arange(batch_size), ground_truth_data]
        #     would be 1
        # - We take the logarithm of this, so that numbers under 1 becomes
        #   negative (in perfect prediction log(1)=0)
        #   - In order to get a positive number to minimize we take the negative
        #     of the result
        # - The mean of this the number we want to minimize
        loss = -prob[torch.arange(batch_size), ground_truth_data].log().mean()

    return loss
