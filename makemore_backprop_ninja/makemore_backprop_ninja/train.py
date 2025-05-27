"""Module to train the model."""

import argparse
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from makemore_backprop_ninja.backprop_helpers.dataclasses import BackpropMode
from makemore_backprop_ninja.backprop_helpers.gradients import (
    attach_gradients,
    compare_gradients,
)
from makemore_backprop_ninja.backprop_helpers.verbose_backprop import (
    verbose_manual_backprop,
)
from makemore_backprop_ninja.data_classes import (
    BatchNormalizationParameters,
    ModelParams,
    OptimizationParams,
)
from makemore_backprop_ninja.evaluation import evaluate
from makemore_backprop_ninja.models import get_explicit_model
from makemore_backprop_ninja.predict import predict_neural_network
from makemore_backprop_ninja.preprocessing import get_dataset
from tqdm import tqdm

from makemore_backprop_ninja import DATASET, DEVICE


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-arguments,too-many-locals,too-complex,too-many-branches,too-many-positional-arguments
def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    batch_normalization_parameters: BatchNormalizationParameters,
    dataset: DATASET,
    optimization_params: Optional[OptimizationParams],
    backprop_mode: BackpropMode = BackpropMode.AUTOMATIC,
    seed: int = 2147483647,
) -> Tuple[torch.Tensor, ...]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model to use
        batch_normalization_parameters (BatchNormalizationParameters):
            Contains the running mean and the running standard deviation
        dataset: DATASET
            Data containing the training and validation set
        optimization_params (Optional[OptimizationParams]): Optimization
            options
        backprop_mode (BackpropMode): The backprop mode to use
        seed (int): The seed for the random number generator

    Returns:
        Tuple[torch.Tensor, ...]: The trained model
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
        logits, intermediate_variables = predict_neural_network(
            model=model,
            input_data=dataset["training_input_data"][idxs],
            batch_normalization_parameters=batch_normalization_parameters,
            training=True,
        )
        intermediate_variables["logits"] = logits
        targets = dataset["training_ground_truth"][idxs]

        if backprop_mode != BackpropMode.VERBOSE:
            loss = F.cross_entropy(logits, targets)
        else:
            # The written out version of the cross entropy
            # NOTE: The logits have shape (batch_size, VOCAB_SIZE)
            #       Taking the max across dim 1 will give the shape
            #       (1, VOCAB_SIZE)
            logits_maxes = logits.max(1, keepdim=True).values
            # Normalize the logits for numerical stability
            normalized_logits = logits - logits_maxes
            counts = normalized_logits.exp()
            # NOTE: With the sum, we go from (batch_size, VOCAB_SIZE) to
            #       (batch_size, 1)
            counts_sum = counts.sum(1, keepdims=True)
            # (1.0/counts_sum) doesn't give the exact values
            counts_sum_inv = counts_sum**-1
            probabilities = counts * counts_sum_inv
            log_probabilities = probabilities.log()
            # The first index picks the row (a batch)
            # For the picked row, the second index picks an element for the
            # first index (a character is picked from the batch)
            # This is equivalent to sparse cross-entropy
            # See note in verbose_manual_backprop for more details
            batch_size = idxs.size(dim=0)
            loss = -log_probabilities[range(batch_size), targets].mean()

            # Add variables to dictionary for better variable handling

            # NOTE: logits_maxes is not used in any calculations of the backprop
            #       However, we still need the variable to compare the gradients
            intermediate_variables["logits_maxes"] = logits_maxes
            # NOTE: normalized_logits is not used in any calculations of the backprop
            #       However, we still need the variable to compare the gradients
            intermediate_variables["normalized_logits"] = normalized_logits
            intermediate_variables["counts"] = counts
            intermediate_variables["counts_sum"] = counts_sum
            intermediate_variables["counts_sum_inv"] = counts_sum_inv
            intermediate_variables["probabilities"] = probabilities
            intermediate_variables["log_probabilities"] = log_probabilities

        # Backward pass
        layered_parameters = model

        # Reset the gradients
        for parameters in layered_parameters:
            parameters.grad = None
        # As we will not do loss.backward() we need to retain the gradients
        for tensor in intermediate_variables.values():
            tensor.retain_grad()

        # Do the back propagation
        if backprop_mode == BackpropMode.VERBOSE:
            gradients = verbose_manual_backprop(
                model=model,
                intermediate_variables=intermediate_variables,
                targets=targets,
                input_data=dataset["training_input_data"][idxs],
            )

        # Always do the  backprop in order to compare
        loss.backward()

        if backprop_mode == BackpropMode.VERBOSE:
            if i % optimization_params.mini_batches_per_data_capture == 0:
                compare_gradients(
                    model=model,
                    intermediate_variables=intermediate_variables,
                    gradients=gradients,
                )
            attach_gradients(model=model, gradients=gradients)

        # Update the weights
        for parameters in layered_parameters:
            parameters.data += (
                -optimization_params.learning_rate(optimization_params.cur_step)
                * parameters.grad
            )

        if i % optimization_params.mini_batches_per_data_capture == 0:
            print(
                f"{optimization_params.cur_step:7d}/"
                f"{optimization_params.n_mini_batches:7d}: "
                f"{loss.item():.4f}"
            )

    # Predict on the whole training set
    training_loss = evaluate(
        model=model,
        input_data=dataset["training_input_data"],
        ground_truth=dataset["training_ground_truth"],
        batch_normalization_parameters=batch_normalization_parameters,
    )
    # Predict on evaluation set
    validation_loss = evaluate(
        model=model,
        input_data=dataset["validation_input_data"],
        ground_truth=dataset["validation_ground_truth"],
        batch_normalization_parameters=batch_normalization_parameters,
    )

    print(f"Final train loss: {training_loss:.3f}")
    print(f"Final validation loss: {validation_loss:.3f}")

    return model


def train(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    batch_normalization_parameters: BatchNormalizationParameters,
    backprop_mode: BackpropMode = BackpropMode.AUTOMATIC,
    seed: int = 2147483647,
) -> Tuple[torch.Tensor, ...]:
    """Train the model.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        batch_normalization_parameters (BatchNormalizationParameters):
            Contains the running mean and the running standard deviation
        backprop_mode (BackpropMode): The backprop mode to use
        seed (int): The seed for the random number generator

    Returns:
        Tuple[torch.Tensor, ...]: The model
    """
    # Obtain the data
    dataset = get_dataset(block_size=model_params.block_size)

    # Obtain the model
    model = get_explicit_model(model_params)

    # Train for one step
    model = train_neural_net_model(
        model=model,
        dataset=dataset,
        batch_normalization_parameters=batch_normalization_parameters,
        optimization_params=optimization_params,
        backprop_mode=backprop_mode,
        seed=seed,
    )

    return model


def train_model(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    backprop_mode: BackpropMode,
) -> None:
    """Train the model.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        backprop_mode (BackpropMode): What backprop mode to use
    """
    # These parameters will be used as batch norm parameters during inference
    # Initialized to zero as the mean and one as std as the initialization of w1
    # and b1 is so that h_pre_activation is roughly gaussian
    batch_normalization_parameters = BatchNormalizationParameters(
        running_mean=torch.zeros(
            (1, model_params.hidden_layer_neurons),
            requires_grad=False,
            device=DEVICE,
        ),
        running_std=torch.ones(
            (1, model_params.hidden_layer_neurons),
            requires_grad=False,
            device=DEVICE,
        ),
    )
    _ = train(
        model_params=model_params,
        optimization_params=optimization_params,
        batch_normalization_parameters=batch_normalization_parameters,
        backprop_mode=backprop_mode,
    )
    print("Training done!")


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
            "Example using batch normalization\n"
            "python3 -m makemore_backprop_ninja.train -m"
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
        "--backprop-mode",
        type=BackpropMode,
        required=False,
        default=BackpropMode.AUTOMATIC,
        choices=list(BackpropMode),
        help="What backprop mode to use",
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
    train_model(
        model_params=model_params,
        optimization_params=optimization_params,
        backprop_mode=args.backprop_mode,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
