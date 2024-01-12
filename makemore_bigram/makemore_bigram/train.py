"""Module to train the model."""

from typing import Tuple

import torch
from makemore_bigram.metrics import calculate_avg_nll_of_neural_net_model
from makemore_bigram.models import get_simple_neural_net
from makemore_bigram.preprocessing import create_training_data, get_padded_data
from makemore_bigram.utils.inference_helper import (
    neural_net_inference_without_interpretation,
)
from makemore_bigram.utils.train_helper import create_bigram_count, create_count_matrix


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
    # The count matrix has dim N_TOKENS, N_TOKENS
    # .sum will create a matrix with dims N_TOKENS, 1
    # Broadcasting rules will copy .sum to be N_TOKENS ->, N_TOKENS
    # If we had keep_dims=False we would get dims 0, N_TOKENS
    # Broadcasting rules would the copy .sum to be N_TOKENS, <- N_TOKENS
    probability_matrix /= probability_matrix.sum(dim=1, keepdim=True)
    return probability_matrix


def train_neural_net_model(
    model: torch.Tensor,
    input_data: Tuple[str, ...],
    epochs: int = 200,
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

    # Create input and ground_truth
    one_hot_input, ground_truth = create_training_data(input_data=input_data)
    # This is equivalent to one_hot_input.shape[0]
    n_examples = ground_truth.nelement()

    for k in range(epochs):
        # Forward pass
        probabilities = neural_net_inference_without_interpretation(
            input_data=one_hot_input, weights=weights
        )
        loss = calculate_avg_nll_of_neural_net_model(
            probabilities=probabilities,
            ground_truth=ground_truth,
            n_examples=n_examples,
        )

        # Backward pass
        # Zero out the gradients
        weights.grad = None
        loss.backward()

        # Update
        assert weights.grad is not None
        weights.data += -learning_rate * weights.grad

        if k % 10 == 0:
            print(f"Epoch={k}, loss={loss.item():.4f}")

    print(f"Final epoch={k}, loss={loss.item():.4f}")

    # Alias
    model = weights
    return model


def get_probability_matrix() -> torch.Tensor:
    """
    Return the (trained) probability tensor for the data.

    Note that this is the same as the trained matrix model.

    Returns:
        torch.Tensor: The probability matrix
    """
    input_data = get_padded_data()
    probability_matrix = train_probability_matrix(input_data=input_data)
    return probability_matrix


def get_neural_net() -> torch.Tensor:
    """
    Return a trained neural net.

    Returns:
        torch.Tensor: The weights of the trained net
    """
    model = get_simple_neural_net()
    padded_data = get_padded_data()
    neural_net = train_neural_net_model(model=model, input_data=padded_data)
    return neural_net
