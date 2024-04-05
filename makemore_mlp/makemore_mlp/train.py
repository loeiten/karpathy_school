"""Module to train the model."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_mlp.data_classes import ModelParams, OptimizationParams, TrainStatistics
from makemore_mlp.inference import predict_neural_network
from makemore_mlp.models import get_model
from makemore_mlp.preprocessing import get_train_validation_and_test_set
from makemore_mlp.visualisation import plot_training


def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    input_training_data: torch.Tensor,
    ground_truth_data: torch.Tensor,
    optimization_params: Optional[OptimizationParams],
    seed: int = 2147483647,
) -> Tuple[Tuple[torch.Tensor], List[float], List[int]]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_training_data (torch.Tensor): The input training data with shape
            (n_samples, block_size).
            This is the data that will be fed into the features
        ground_truth_data (torch.Tensor): The correct prediction for the input
            (the correct labels)
        optimization_params (Optional[OptimizationParams]): Optimization
            options
        seed (int): The seed for the random number generator

    Returns:
        Tuple[torch.Tensor, ...]: The trained model
        List[float]: The loss of each step
        List[int]: The step
    """
    if optimization_params is None:
        optimization_params = OptimizationParams()

    # Make it possible to train
    for parameters in model:
        parameters.requires_grad = True

    g = torch.Generator().manual_seed(seed)

    step_list = []
    loss_list = []
    # NOTE: It's better to take a lot of steps in the approximate direction of
    #       the true gradient than it is to take one big step in the direction
    #       of the true gradient
    for i in range(optimization_params.mini_batches_per_iteration):
        # Mini batch constructor
        n_samples = input_training_data.shape[0]
        idxs = torch.randint(
            low=0, high=n_samples, size=(optimization_params.batch_size,), generator=g
        )

        # Update cur_mini_batch number
        optimization_params.cur_mini_batch += 1

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
        #     -prob[torch.arange(optimization_params.batch_size), ground_truth_data]
        #     .log()
        #     .mean()
        # )

        # However, it's much more efficient both mathematically and in terms of
        # backprop
        # It's also numerically better behaved
        loss = F.cross_entropy(logits, ground_truth_data[idxs])

        # Append loss and iteration
        loss_list.append(loss.item())
        step_list.append(i)

        # Backward pass

        # Reset the gradients
        for parameters in model:
            parameters.grad = None

        loss.backward()

        # Update the weights
        for parameters in model:
            parameters.data += (
                -optimization_params.learning_rate(optimization_params.cur_mini_batch)
                * parameters.grad
            )

    return model, loss_list, step_list


def main() -> None:
    """Train and plot the model."""
    model_params = ModelParams()

    # Obtain the data
    (
        train_input,
        train_output,
        validate_input,
        validate_output,
        _,  # test_input,
        _,  # test_output,
    ) = get_train_validation_and_test_set(block_size=model_params.block_size)

    # Obtain the model
    model = get_model(
        block_size=model_params.block_size,
        embedding_size=model_params.embedding_size,
        hidden_layer_neurons=model_params.hidden_layer_neurons,
    )

    optimization_params = OptimizationParams()
    train_statistics = TrainStatistics()

    cur_step = 0

    for i in range(optimization_params.n_iterations):
        # Train for one step
        model, loss, step = train_neural_net_model(
            model=model,
            input_training_data=train_input,
            ground_truth_data=train_output,
            optimization_params=optimization_params,
            seed=i,  # Change the seed in order not to train on the same data
        )

        # Save statistics
        train_statistics.train_loss.extend(loss)
        train_statistics.train_step.extend([s + cur_step for s in step])
        cur_step = train_statistics.train_step[-1]

        # Predict on evaluation set
        logits = predict_neural_network(model=model, input_data=validate_input)
        cur_eval_loss = F.cross_entropy(logits, validate_output)
        train_statistics.eval_loss.append(cur_eval_loss.item())
        train_statistics.eval_step.append(train_statistics.train_step[-1])

    print(f"Final train loss: {train_statistics.train_loss[-1]:.3f}")
    print(f"Final validation loss: {train_statistics.eval_loss[-1]:.3f}")

    plot_training(train_statistics=train_statistics)


if __name__ == "__main__":
    main()
