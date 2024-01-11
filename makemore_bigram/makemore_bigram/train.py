"""Module to train the model."""

from typing import Tuple

import torch
from makemore_bigram.inference import neural_net_inference_without_interpretation
from makemore_bigram.metrics import calculate_avg_nll_of_matrix_model
from makemore_bigram.preprocessing import get_padded_data
from makemore_bigram.utils.train_helper import create_bigram_count, create_count_matrix


def train_neural_net_model(
    model: torch.Tensor,
    input_data: Tuple[str, ...],
    epochs: int = 100,
    learning_rate: float = 50.0,
) -> torch.Tensor:
    """Train the neural net model.

    Args:
        model (torch.Tensor): The model (weights) to use
        input_data (Tuple[str, ...]): The input data
        epochs (int, optional): Number of epochs. Defaults to 100.
        learning_rate (float, optional): The learning rate. Defaults to 50.0.

    Returns:
        torch.Tensor: The trained model
    """
    # Alias
    weights = model

    for k in range(epochs):
        # Forward pass
        probabilities = neural_net_inference_without_interpretation(
            input_data=input_data, weights=weights
        )
        loss = calculate_avg_nll_of_matrix_model(data=input_data, model=probabilities)

        # Backward pass
        # Zero out the gradients
        weights.zero()
        loss.backward()

        # Update
        weights.data += -learning_rate * weights.grad

        if k % 10 == 0:
            print(f"Epoch={k}, loss={loss.item():.4f}")

    # Alias
    model = weights
    return model


def train_probability_matrix(input_data: Tuple[str, ...]) -> torch.Tensor:
    """
    Create a probability matrix from the input.

    Note that there is no input for the model as the model is created directly
    from the input_data.

    Args:
        input_data (Tuple[str, ...]): The input data

    Returns:
        torch.Tensor: The probability matrix
    """
    # Obtain the count matrix
    bigram_count = create_bigram_count(padded_data=input_data)
    count_matrix = create_count_matrix(bigram_dict=bigram_count)

    # We add one to the probability matrix to avoid dividing by zero
    probability_matrix = (count_matrix + 1).float()

    # We take the sum along the rows (axis=1)
    # We have keep_dims=True due to the broadcasting rules
    # The count matrix has dim 27, 27
    # .sum will create a matrix with dims 27, 1
    # Broadcasting rules will copy .sum to be 27 ->, 27
    # If we had keep_dims=False we would get dims 0, 27
    # Broadcasting rules would the copy .sum to be 27, <- 27
    probability_matrix /= probability_matrix.sum(dim=1, keepdim=True)
    return probability_matrix


def get_probability_matrix() -> torch.Tensor:
    """
    Return the probability tensor for the data.

    Note that this is the same as the trained matrix model.

    Returns:
        torch.Tensor: The probability matrix
    """
    input_data = get_padded_data()
    probability_matrix = train_probability_matrix(input_data=input_data)
    return probability_matrix


if __name__ == "__main__":
    from makemore_bigram.models import get_simple_neural_net

    neural_net = get_simple_neural_net()
    names_ = get_padded_data()
