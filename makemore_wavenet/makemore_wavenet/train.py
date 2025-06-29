"""Module to train the model."""

import argparse
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_wavenet.data_classes import (
    ModelParams,
    OptimizationParams,
    TrainStatistics,
    ModelType,
)
from makemore_wavenet.evaluation import evaluate
from makemore_wavenet.models import (
    get_vanilla_model,
    get_original_12k,
    get_context_8_22k,
)
from makemore_wavenet.module import Sequential
from makemore_wavenet.preprocessing import get_dataset
from makemore_wavenet.visualisation import plot_training
from tqdm import tqdm

from makemore_wavenet import DATASET, DEVICE


# Reducing the number of locals here will penalize the didactical purpose
def train_neural_net_model(
    model: Sequential,
    dataset: DATASET,
    optimization_params: Optional[OptimizationParams],
    seed: int = 2147483647,
    train_statistics: Optional[TrainStatistics] = None,
) -> Sequential:
    """Train the neural net model.

    Args:
        model (Tuple[Module, ...]): The model to use
        dataset: DATASET
            Data containing the training and validation set
        optimization_params (Optional[OptimizationParams]): Optimization
            options
        seed (int): The seed for the random number generator
        train_statistics (Optional[TrainStatistics]): Class to capture the
            statistics of the training job

    Raises:
        TypeError: If wrong model type is given

    Returns:
        Sequential: The trained model
    """
    if optimization_params is None:
        optimization_params = OptimizationParams()

    g = torch.Generator(device=DEVICE).manual_seed(seed)

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
            low=0,
            high=n_samples,
            size=(optimization_params.batch_size,),
            generator=g,
            device=DEVICE,
        )

        # Forward pass
        # NOTE: training_input_data has dimension (n_samples, block_size)
        #       training_input_data[idxs] selects batch_size samples from the
        #       training data
        #       The size of training_input_data[idxs] is therefore
        #       (batch_size, block_size)
        logits = model(dataset["training_input_data"][idxs])
        loss = F.cross_entropy(logits, dataset["training_ground_truth"][idxs])

        # Append loss and iteration
        if train_statistics is not None:
            train_statistics.training_loss.append(loss.item())
            train_statistics.training_step.append(optimization_params.cur_step)

        # Backward pass
        # Reset the gradients
        for parameters in model.parameters():
            parameters.grad = None
        loss.backward()

        # Update the weights
        for parameters in model.parameters():
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
                )
                train_statistics.eval_training_loss.append(cur_training_loss)
                train_statistics.eval_training_step.append(optimization_params.cur_step)
                # Predict on evaluation set
                cur_validation_loss = evaluate(
                    model=model,
                    input_data=dataset["validation_input_data"],
                    ground_truth=dataset["validation_ground_truth"],
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
    model_type: ModelType = ModelType.NONE,
) -> Tuple[Sequential, TrainStatistics]:
    """Train the model.

    Raises:
        ValueError: If the model_type is non-existent

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        seed (int): The seed for the random number generator
        model_type (ModelType): What pre-canned model to use (if any).
            If set to something else than ModelType.NONE the model_params will
            be overwritten

    Returns:
        Sequential: The model
        TrainStatistics: Statistics from the training
    """
    # Obtain the model (need to be first at it may alter model params)
    if model_type == ModelType.NONE:
        model = get_vanilla_model(model_params=model_params)
    elif model_type == ModelType.ORIGINAL_12K:
        model = get_original_12k(model_params=model_params)
    elif model_type == ModelType.CONTEXT_8_22K:
        model = get_context_8_22k(model_params=model_params)
    else:
        ValueError(f"No model with type {model_type}")

    # Obtain the data
    dataset = get_dataset(block_size=model_params.block_size)

    train_statistics = TrainStatistics()

    # Train for one step
    model = train_neural_net_model(
        model=model,
        dataset=dataset,
        optimization_params=optimization_params,
        seed=seed,
        train_statistics=train_statistics,
    )

    print(f"Final train loss: {train_statistics.eval_training_loss[-1]:.3f}")
    print(f"Final validation loss: {train_statistics.eval_validation_loss[-1]:.3f}")

    return model, train_statistics


def train_and_plot(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    model_type: ModelType = ModelType.NONE,
) -> None:
    """Train the model and plot the statistics.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        model_type (ModelType): What pre-canned model to use (if any).
            If set to something else than ModelType.NONE the model_params will
            be overwritten
    """
    _, train_statistics = train(
        model_params=model_params,
        optimization_params=optimization_params,
        model_type=model_type,
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
        epilog=("Example using batch normalization\npython3 -m makemore_wavenet.train"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_model_params = ModelParams()
    parser.add_argument(
        "-t",
        "--model-type",
        type=ModelType,
        required=False,
        default=ModelType.NONE,
        choices=list(ModelType),
        help=(
            "The pre-defined models. "
            "If selected, they will overwrite other parameters set in the input."
        ),
    )
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
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
