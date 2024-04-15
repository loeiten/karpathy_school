"""Module to train the model."""

import argparse
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_mlp.data_classes import ModelParams, OptimizationParams, TrainStatistics
from makemore_mlp.evaluation import evaluate
from makemore_mlp.inference import predict_neural_network
from makemore_mlp.models import get_model
from makemore_mlp.preprocessing import get_train_validation_and_test_set
from makemore_mlp.visualisation import plot_training
from tqdm import tqdm


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
    for i in tqdm(
        range(optimization_params.mini_batches_per_iteration),
        desc="Mini-batch per iteration",
        leave=False,
    ):
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


def train_and_plot(
    model_params: ModelParams, optimization_params: OptimizationParams
) -> None:
    """Train the model and plot the statistics.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
    """
    # Obtain the data
    (
        training_input,
        training_output,
        validation_input,
        validation_output,
        _,  # test_input,
        _,  # test_output,
    ) = get_train_validation_and_test_set(block_size=model_params.block_size)

    # Obtain the model
    model = get_model(
        block_size=model_params.block_size,
        embedding_size=model_params.embedding_size,
        hidden_layer_neurons=model_params.hidden_layer_neurons,
    )

    train_statistics = TrainStatistics()

    cur_step = 0

    for i in tqdm(range(optimization_params.n_iterations), desc="Iteration"):
        # Train for one step
        model, loss, step = train_neural_net_model(
            model=model,
            input_training_data=training_input,
            ground_truth_data=training_output,
            optimization_params=optimization_params,
            seed=i,  # Change the seed in order not to train on the same data
        )

        # Save statistics
        train_statistics.training_loss.extend(loss)
        train_statistics.training_step.extend([s + cur_step for s in step])
        cur_step = train_statistics.training_step[-1]

        # Predict on the whole training set
        cur_training_loss = evaluate(
            model=model, input_data=training_input, ground_truth=training_output
        )
        train_statistics.eval_training_loss.append(cur_training_loss)
        train_statistics.eval_training_step.append(train_statistics.training_step[-1])
        # Predict on evaluation set
        cur_validation_loss = evaluate(
            model=model, input_data=validation_input, ground_truth=validation_output
        )
        train_statistics.eval_validation_loss.append(cur_validation_loss)
        train_statistics.eval_validation_step.append(train_statistics.training_step[-1])

    print(f"Final train loss: {train_statistics.eval_training_loss[-1]:.3f}")
    print(f"Final validation loss: {train_statistics.eval_validation_loss[-1]:.3f}")

    plot_training(train_statistics=train_statistics)


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    """Parse the arguments.

    Args:
        sys_args (List[str]): The system arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a model and plot its contents.",
        epilog=(
            "Increase the size of the hidden layer and train for longer\n"
            "python3 -m makemore_mlp.train -l 300 -t 120000 -m 1000\n\n"
            "As we're underfitting the above we suspect that the embedding "
            "size is the bottleneck\n"
            "python3 -m makemore_mlp.train -l 200 -e 10 -t 200000 -m 1000\n\n"
            "Training for longer seem to be a good way to decrease the loss"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_model_params = ModelParams()
    parser.add_argument(
        "-s",
        "--block-size",
        type=int,
        required=False,
        default=default_model_params.block_size,
        help=(
            "Number of input features to the network. "
            "This is how many characters we are considering simultaneously, "
            "aka. the context length"
        ),
    )
    parser.add_argument(
        "-e",
        "--embedding-size",
        type=int,
        required=False,
        default=default_model_params.embedding_size,
        help="The size of the embedding space",
    )
    parser.add_argument(
        "-l",
        "--hidden-layer-neurons",
        type=int,
        required=False,
        default=default_model_params.hidden_layer_neurons,
        help="Number of neurons for the hidden layer",
    )

    default_optimization_params = OptimizationParams()
    parser.add_argument(
        "-t",
        "--total-mini-batches",
        type=int,
        required=False,
        default=default_optimization_params.total_mini_batches,
        help="Total number of mini batches to train on",
    )
    parser.add_argument(
        "-m",
        "--mini-batches-per-iteration",
        type=int,
        required=False,
        default=default_optimization_params.mini_batches_per_iteration,
        help="Number of mini batches to run for each call to the training function",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=default_optimization_params.batch_size,
        help="Number of examples per batch",
    )

    args = parser.parse_args(sys_args)
    return args


def main(sys_args: List[str]):
    """Parse the arguments and run train_and_plot.

    Args:
        sys_args (List[str]): The system arguments
    """
    args = parse_args(sys_args)
    model_params = ModelParams(
        block_size=args.block_size,
        embedding_size=args.embedding_size,
        hidden_layer_neurons=args.hidden_layer_neurons,
    )
    optimization_params = OptimizationParams(
        total_mini_batches=args.total_mini_batches,
        mini_batches_per_iteration=args.mini_batches_per_iteration,
        batch_size=args.batch_size,
    )
    train_and_plot(model_params=model_params, optimization_params=optimization_params)


if __name__ == "__main__":
    main(sys.argv[1:])
