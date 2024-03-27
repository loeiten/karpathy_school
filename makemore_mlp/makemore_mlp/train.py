"""Module to train the model."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_mlp.inference import predict_neural_network
from makemore_mlp.options import ModelOptions


def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    input_training_data: torch.Tensor,
    ground_truth_data: torch.Tensor,
    model_options: Optional[ModelOptions],
) -> Tuple[Tuple[torch.Tensor], List[float], List[int]]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_training_data (torch.Tensor): The input training data with shape
            (n_samples, block_size).
            This is the data that will be fed into the features
        ground_truth_data (torch.Tensor): The correct prediction for the input
            (the correct labels)
        model_options (Optional[ModelOptions]): Model option

    Returns:
        Tuple[torch.Tensor, ...]: The trained model
        List[float]: The log10 of the loss for each step
        List[int]: The step
    """
    if model_options is None:
        model_options = ModelOptions()

    # Make it possible to train
    for parameters in model:
        parameters.requires_grad = True

    step = []
    loss_log_10 = []
    # NOTE: It's better to take a lot of steps in the approximate direction of
    #       the true gradient than it is to take one big step in the direction
    #       of the true gradient
    for i in range(model_options.n_mini_batches):
        # Mini batch constructor
        n_samples = input_training_data.shape[0]
        idxs = torch.randint(low=0, high=n_samples, size=model_options.batch_size)

        # NOTE: input_training_data has dimension (n_samples, block_size)
        #       input_training_data[idxs] selects batch_size samples from the
        #       training data
        #       The size of input_training_data[idxs] is therefore
        #       (batch_size, block_size)
        input_data = input_training_data[idxs]

        # Forward pass
        logits = predict_neural_network(model=model, input_data=input_data)

        # NOTE: We could have the following implementation:
        # # Get the fake counts (like in makemore_bigram)
        # counts = logits.exp()
        # # Normalize to probability
        # prob = counts/counts.sum(1, keepdim=True)
        # # Negative loss likelihood:
        # # - For each of the example in the batch: torch.arange(batch_size)
        # #   - Select the probability of getting the ground truth:
        # #     prob[torch.arange(batch_size), ground_truth_data]
        # #     The shape of this will be the batch_size
        # #     In a model that perfectly predicts the characters all entries in
        # #     prob[torch.arange(batch_size), ground_truth_data]
        # #     would be 1
        # # - We take the logarithm of this, so that numbers under 1 becomes
        # #   negative (in perfect prediction log(1)=0)
        # #   - In order to get a positive number to minimize we take the
        # #     negative
        # #     of the result
        # # - The mean of this the number we want to minimize
        # loss = (
        #     -prob[torch.arange(model_options.batch_size), ground_truth_data]
        #     .log()
        #     .mean()
        # )

        # However, it's much more efficient both mathematically and in terms of
        # backprop
        # It's also numerically better behaved
        loss = F.cross_entropy(logits, ground_truth_data)

        # Append loss and iteration
        loss_log_10.append(loss.log10().item())
        step.append(i)

        # Backward pass

        # Reset the gradients
        for parameters in model:
            parameters.grad = None

        loss.backward()

        # Update the weights
        for parameters in model:
            parameters.data += -model_options.learning_rate * parameters.grad

    return model, loss_log_10, step
