"""Module to train the model."""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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
# pylint: disable-next=too-many-arguments,too-many-locals,too-complex,too-many-branches
def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    batch_normalization_parameters: BatchNormalizationParameters,
    dataset: DATASET,
    optimization_params: Optional[OptimizationParams],
    use_functional: bool = True,
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
        use_functional (bool): Whether or not to use the functional version of
            the cross entropy.
            If False, the hand-written version will be used
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
        if use_functional:
            loss = F.cross_entropy(logits, dataset["training_ground_truth"][idxs])
        else:
            # The written out version of the cross entropy
            logits_maxes = logits.max(1, keepdim=True).values
            # Normalize the logits for numerical stability
            normalized_logits = logits - logits_maxes
            counts = normalized_logits.exp()
            counts_sum = counts.sum(1, keepdims=True)
            # (1.0/counts_sum) doesn't give the exact values
            counts_sum_inv = counts_sum**-1
            probabilities = counts * counts_sum_inv
            log_probabilities = probabilities.log()
            batch_size = idxs.size(dim=0)
            loss = -log_probabilities[
                range(batch_size), dataset["training_ground_truth"][idxs]
            ].mean()

            # Add variables to dictionary for better variable handling
            intermediate_variables["logits_maxes"] = logits_maxes
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
        gradients = manual_backprop(
            model=model, intermediate_variables=intermediate_variables
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


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-locals,too-many-statements
def manual_backprop(
    model: Tuple[torch.Tensor, ...], intermediate_variables: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Do the manual back propagation, and set the gradients to the parameters.

    Args:
        model (Tuple[torch.Tensor,...]): The weights of the model
        intermediate_variables (Dict[str, torch.Tensor]): The intermediate
            variables (i.e. those which are not part of model parameters).

    Returns:
        A map of the gradients
    """
    # Alias for the model weights
    (
        c,
        w1,
        b1,
        w2,
        b2,
        batch_normalization_gain,
        batch_normalization_bias,
    ) = model
    # Intermediate variables from predict
    embedding = intermediate_variables["embedding"]
    concatenated_embedding = intermediate_variables["concatenated_embedding"]
    h_pre_batch_norm = intermediate_variables["h_pre_batch_norm"]
    batch_normalization_mean = intermediate_variables["batch_normalization_mean"]
    batch_normalization_diff = intermediate_variables["batch_normalization_diff"]
    batch_normalization_diff_squared = intermediate_variables[
        "batch_normalization_diff_squared"
    ]
    batch_normalization_var = intermediate_variables["batch_normalization_var"]
    inv_batch_normalization_std = intermediate_variables["inv_batch_normalization_std"]
    batch_normalization_raw = intermediate_variables["batch_normalization_raw"]
    h_pre_activation = intermediate_variables["h_pre_activation"]
    h = intermediate_variables["h"]
    # Intermediate variables from loss
    logits = intermediate_variables["logits"]
    logits_maxes = intermediate_variables["logits_maxes"]
    normalized_logits = intermediate_variables["normalized_logits"]
    counts = intermediate_variables["counts"]
    counts_sum = intermediate_variables["counts_sum"]
    counts_sum_inv = intermediate_variables["counts_sum_inv"]
    probabilities = intermediate_variables["probabilities"]
    log_probabilities = intermediate_variables["log_probabilities"]

    # Calculate the gradients
    # Calculate the derivatives of the cross entropy
    dl_d_log_probabilities = torch.zeros_like(log_probabilities)
    dl_d_probabilities = torch.zeros_like(probabilities)
    dl_d_counts_sum_inv = torch.zeros_like(counts_sum_inv)
    dl_d_counts_sum = torch.zeros_like(counts_sum)
    dl_d_counts = torch.zeros_like(counts)
    dl_d_normalized_logits = torch.zeros_like(normalized_logits)
    dl_d_logits_maxes = torch.zeros_like(logits_maxes)
    dl_d_logits = torch.zeros_like(logits)
    # Calculate the derivatives of the second layer
    dl_dh = torch.zeros_like(h)
    dl_dh_pre_activation = torch.zeros_like(h_pre_activation)
    dl_dw2 = torch.zeros_like(w2)
    dl_db2 = torch.zeros_like(b2)
    # Calculate the derivatives of the batch norm layer (of the first layer)
    dl_dbatch_normalization_raw = torch.zeros_like(batch_normalization_raw)
    dl_dinv_batch_normalization_std = torch.zeros_like(inv_batch_normalization_std)
    dl_dbatch_normalization_var = torch.zeros_like(batch_normalization_var)
    dl_dbatch_normalization_diff_squared = torch.zeros_like(
        batch_normalization_diff_squared
    )
    dl_dbatch_normalization_diff = torch.zeros_like(batch_normalization_diff)
    dl_dbatch_normalization_mean = torch.zeros_like(batch_normalization_mean)
    dl_dh_pre_batch_norm = torch.zeros_like(h_pre_batch_norm)
    dl_dbatch_normalization_gain = torch.zeros_like(batch_normalization_gain)
    dl_dbatch_normalization_bias = torch.zeros_like(batch_normalization_bias)
    # Calculate the derivatives of the first layer
    dl_dw1 = torch.zeros_like(w1)
    dl_db1 = torch.zeros_like(b1)
    # Calculate the derivatives of the embedding layer
    dl_dconcatenated_embedding = torch.zeros_like(concatenated_embedding)
    dl_dembedding = torch.zeros_like(embedding)
    dl_dc = torch.zeros_like(c)

    gradients: Dict[str, torch.Tensor] = {}
    gradients["dl_d_log_probabilities"] = dl_d_log_probabilities
    gradients["dl_d_probabilities"] = dl_d_probabilities
    gradients["dl_d_counts_sum_inv"] = dl_d_counts_sum_inv
    gradients["dl_d_counts_sum"] = dl_d_counts_sum
    gradients["dl_d_counts"] = dl_d_counts
    gradients["dl_d_normalized_logits"] = dl_d_normalized_logits
    gradients["dl_d_logits_maxes"] = dl_d_logits_maxes
    gradients["dl_d_logits"] = dl_d_logits
    gradients["dl_dh"] = dl_dh
    gradients["dl_dh_pre_activation"] = dl_dh_pre_activation
    gradients["dl_dw2"] = dl_dw2
    gradients["dl_db2"] = dl_db2
    gradients["dl_dbatch_normalization_raw"] = dl_dbatch_normalization_raw
    gradients["dl_dinv_batch_normalization_std"] = dl_dinv_batch_normalization_std
    gradients["dl_dbatch_normalization_var"] = dl_dbatch_normalization_var
    gradients["dl_dbatch_normalization_diff_squared"] = (
        dl_dbatch_normalization_diff_squared
    )
    gradients["dl_dbatch_normalization_diff"] = dl_dbatch_normalization_diff
    gradients["dl_dbatch_normalization_mean"] = dl_dbatch_normalization_mean
    gradients["dl_dh_pre_batch_norm"] = dl_dh_pre_batch_norm
    gradients["dl_dbatch_normalization_gain"] = dl_dbatch_normalization_gain
    gradients["dl_dbatch_normalization_bias"] = dl_dbatch_normalization_bias
    gradients["dl_dw1"] = dl_dw1
    gradients["dl_db1"] = dl_db1
    gradients["dl_dconcatenated_embedding"] = dl_dconcatenated_embedding
    gradients["dl_dembedding"] = dl_dembedding
    gradients["dl_dc"] = dl_dc

    return gradients


def attach_gradients(
    model: Tuple[torch.Tensor, ...], gradients: Dict[str, torch.Tensor]
) -> None:
    """Attach gradients from the manual back-propagation to the model.

    Args:
        model (Tuple[torch.Tensor,...]): Model weights
        gradients (Dict[str, torch.Tensor]): The gradients
    """
    # Alias for the model weights
    (
        c,
        w1,
        b1,
        w2,
        b2,
        batch_normalization_gain,
        batch_normalization_bias,
    ) = model
    # Attach the gradients to the variables
    # NOTE: Only the gradients of the model variables are needed.
    #       The gradients of the intermediate variables are only needed for
    #       calculating the gradients of the model weights
    # Gradients of the second layer
    w2.grad = gradients["dl_dw2"]
    b2.grad = gradients["dl_db2"]
    # Gradients of the batch norm layer
    batch_normalization_gain.grad = gradients["dl_dbatch_normalization_gain"]
    batch_normalization_bias.grad = gradients["dl_dbatch_normalization_bias"]
    # Gradients of the first layer
    w1.grad = gradients["dl_dw1"]
    b1.grad = gradients["dl_db1"]
    # Gradients of the embedding layer
    c.grad = gradients["dl_dc"]


def compare_manual_gradient_with_real(
    name: str, manually_calculated: torch.Tensor, tensor: torch.Tensor
) -> None:
    """
    Compare the manually calculated gradient with the one calculated using autograd.

    Args:
        name (str): Name of the tensor
        manually_calculated (torch.Tensor): The manually calculated gradient
        tensor (torch.Tensor): The tensor to check
    """
    exact = torch.all(manually_calculated == tensor.grad).item()
    approximate = torch.allclose(manually_calculated, tensor.grad)
    max_diff = (manually_calculated - tensor.grad).abs().max().item()
    print(
        f"{name:15s} | "
        f"exact: {str(exact):5s} | "
        f"approximate {str(approximate):5s} | "
        f"max difference: {max_diff}"
    )


def train(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    batch_normalization_parameters: BatchNormalizationParameters,
    use_functional: bool = True,
    seed: int = 2147483647,
) -> Tuple[torch.Tensor, ...]:
    """Train the model.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        batch_normalization_parameters (BatchNormalizationParameters):
            Contains the running mean and the running standard deviation
        use_functional (bool): Whether or not to use the functional version of
            the cross entropy.
            If False, the hand-written version will be used
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
        use_functional=use_functional,
        seed=seed,
    )

    return model


def train_model(
    model_params: ModelParams,
    optimization_params: OptimizationParams,
    use_functional: bool,
) -> None:
    """Train the model.

    Args:
        model_params (ModelParams): The model parameters
        optimization_params (OptimizationParams): The optimization parameters
        use_functional (bool): Whether or not to use the functional version of
            the cross entropy.
            If False, the hand-written version will be used
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
        use_functional=use_functional,
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
        epilog=("Example using batch normalization\npython3 -m makemore_agb.train -m"),
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
        "-u",
        "--use-functional",
        type=bool,
        required=False,
        default=True,
        help="Whether or not to use the functional version of the cross entropy.",
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
        use_functional=args.use_functional,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
