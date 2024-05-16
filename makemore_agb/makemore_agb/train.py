"""Module to train the model."""

import argparse
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_agb.data_classes import ModelParams, OptimizationParams, TrainStatistics
from makemore_agb.evaluation import evaluate
from makemore_agb.models import get_model
from makemore_agb.predict import predict_neural_network
from makemore_agb.preprocessing import get_dataset
from makemore_agb.visualisation import plot_training
from tqdm import tqdm

from makemore_agb import DATASET


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable=too-many-arguments
def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    dataset: DATASET,
    optimization_params: Optional[OptimizationParams],
    seed: int = 2147483647,
    train_statistics: Optional[TrainStatistics] = None,
    batch_normalize: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        dataset: DATASET
            Data containing the training and validation set
        optimization_params (Optional[OptimizationParams]): Optimization
            options
        seed (int): The seed for the random number generator
        train_statistics (Optional[TrainStatistics]): Class to capture the
            statistics of the training job
        batch_normalize (bool): Whether or not to use batch normalization

    Returns:
        Tuple[torch.Tensor, ...]: The trained model
    """
    if optimization_params is None:
        optimization_params = OptimizationParams()

    # Make it possible to train
    for parameters in model:
        parameters.requires_grad = True

    g = torch.Generator().manual_seed(seed)

    # NOTE: It's better to take a lot of steps in the approximate direction of
    #       the true gradient than it is to take one big step in the direction
    #       of the true gradient
    for i in tqdm(
        range(optimization_params.n_mini_batches),
        desc="Mini batch",
    ):
        optimization_params.cur_step += 1
        # Mini batch constructor
        n_samples = dataset["training_input_data"].shape[0]
        idxs = torch.randint(
            low=0, high=n_samples, size=(optimization_params.batch_size,), generator=g
        )

        # Forward pass
        # NOTE: training_input_data has dimension (n_samples, block_size)
        #       training_input_data[idxs] selects batch_size samples from the
        #       training data
        #       The size of training_input_data[idxs] is therefore
        #       (batch_size, block_size)
        # Note the [0] as predict always returns a tuple
        logits = predict_neural_network(
            model=model,
            input_data=dataset["training_input_data"][idxs],
            batch_normalize=batch_normalize,
            training=True,
        )[0]
        loss = F.cross_entropy(logits, dataset["training_ground_truth"][idxs])

        # Append loss and iteration
        if train_statistics is not None:
            train_statistics.training_loss.append(loss.item())
            train_statistics.training_step.append(optimization_params.cur_step)

        # Backward pass
        # Reset the gradients
        for parameters in model:
            parameters.grad = None
        loss.backward()

        # Update the weights
        for parameters in model:
            parameters.data += (
                -optimization_params.learning_rate(optimization_params.cur_step)
                * parameters.grad
            )

        if i % optimization_params.mini_batches_per_data_capture == 0:
            if train_statistics is not None:
                # Predict on the whole training set
                cur_training_loss = evaluate(
                    model=model,
                    input_data=dataset["training_input_data"],
                    ground_truth=dataset["training_ground_truth"],
                    batch_normalize=batch_normalize,
                )
                train_statistics.eval_training_loss.append(cur_training_loss)
                train_statistics.eval_training_step.append(optimization_params.cur_step)
                # Predict on evaluation set
                cur_validation_loss = evaluate(
                    model=model,
                    input_data=dataset["validation_input_data"],
                    ground_truth=dataset["validation_ground_truth"],
                    batch_normalize=batch_normalize,
                )
                train_statistics.eval_validation_loss.append(cur_validation_loss)
                train_statistics.eval_validation_step.append(
                    optimization_params.cur_step
                )

            print(
                f"{optimization_params.cur_step:7d}/"
                f"{optimization_params.n_mini_batches:7d}: "
                f"{loss.item():.4f}"
            )

    return model


def train(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    seed: int = 2147483647,
    batch_normalize: bool = False,
) -> Tuple[Tuple[torch.Tensor, ...], TrainStatistics]:
    """Train the model.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        seed (int): The seed for the random number generator
        batch_normalize (bool): Whether or not to use batch normalization

    Returns:
        Tuple[torch.Tensor, ...]: The model
        TrainStatistics: Statistics from the training
    """
    # Obtain the data
    dataset = get_dataset(block_size=model_params.block_size)

    # Obtain the model
    model = get_model(
        block_size=model_params.block_size,
        embedding_size=model_params.embedding_size,
        hidden_layer_neurons=model_params.hidden_layer_neurons,
    )

    train_statistics = TrainStatistics()

    # Train for one step
    model = train_neural_net_model(
        model=model,
        dataset=dataset,
        optimization_params=optimization_params,
        seed=seed,
        train_statistics=train_statistics,
        batch_normalize=batch_normalize,
    )

    print(f"Final train loss: {train_statistics.eval_training_loss[-1]:.3f}")
    print(f"Final validation loss: {train_statistics.eval_validation_loss[-1]:.3f}")

    return model, train_statistics


def train_and_plot(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    batch_normalize=False,
) -> None:
    """Train the model and plot the statistics.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        batch_normalize (bool): Whether or not to use batch normalization
    """
    _, train_statistics = train(
        model_params=model_params,
        optimization_params=optimization_params,
        batch_normalize=batch_normalize,
    )

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
            "python3 -m makemore_agb.train -l 300 -t 120000 -m 1000\n\n"
            "As we're underfitting the above we suspect that the embedding "
            "size is the bottleneck\n"
            "python3 -m makemore_agb.train -l 200 -e 10 -t 200000 -m 1000\n\n"
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
        "-n",
        "--n-mini-batches",
        type=int,
        required=False,
        default=default_optimization_params.n_mini_batches,
        help="Total number of mini batches to train on",
    )
    parser.add_argument(
        "-c",
        "--mini-batches-per-data-capture",
        type=int,
        required=False,
        default=default_optimization_params.mini_batches_per_data_capture,
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
    parser.add_argument(
        "-m",
        "--batch-normalization",
        type=bool,
        required=False,
        default=True,
        help="Whether or not to use batch normalization",
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
        n_mini_batches=args.n_mini_batches,
        mini_batches_per_data_capture=args.mini_batches_per_data_capture,
        batch_size=args.batch_size,
    )
    train_and_plot(
        model_params=model_params,
        optimization_params=optimization_params,
        batch_normalize=args.batch_normalize,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
