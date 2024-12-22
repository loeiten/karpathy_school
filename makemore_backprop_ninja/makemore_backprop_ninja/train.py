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
        targets = dataset["training_ground_truth"][idxs]
        if use_functional:
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
            # See note in manual_backprop for more details
            batch_size = idxs.size(dim=0)
            loss = -log_probabilities[range(batch_size), targets].mean()

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
            model=model, intermediate_variables=intermediate_variables, targets=targets
        )
        # Do the actual backprop in order to compare
        loss.backward()
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


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-locals,too-many-statements
def manual_backprop(
    model: Tuple[torch.Tensor, ...],
    intermediate_variables: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Do the manual back propagation, and set the gradients to the parameters.

    Args:
        model (Tuple[torch.Tensor,...]): The weights of the model
        intermediate_variables (Dict[str, torch.Tensor]): The intermediate
            variables (i.e. those which are not part of model parameters).
        targets(torch.Tensor): The targets
            Needed to compute the log_prob gradients

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
    #
    # The cross-entropy between two probability distributions p and q measures
    # the average number of bits needed to identify an event drawn from the set
    # when the coding scheme used for the set is optimized for an estimated
    # probability distribution q, rather than the true distribution p.
    # It reads
    # H(p,q) = sum_x p(x) * log(q(x))
    # https://en.wikipedia.org/wiki/Cross-entropy
    #
    # In our case the true distribution is a one-hot encoding
    # This means that only one of the characters in the vocabulary (classes)
    # have the probability of 1, and the rest have probability of 0
    # Because of this we can use the sparse entropy definition used by PyTorch
    # https://pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    # Note that in knowledge distillation we use the outputs of the teacher
    # model as the target for the student model
    # In this case we can no longer use the torch.nn.CrossEntropyLoss, and we
    # need to use a custom implementation like
    # https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
    #
    # Note that the loss function gives a loss for each batch
    # These batch are then reduced (default using mean)
    # In other words L : R^{N x C} => R
    # Where N is the batch size and C is the number of classes possible to
    # predict
    # Furthermore, the "gradient" is a mapping from f : R => R^{N x C}
    # Each row contains a batch, and each column describes a possible class
    # We want to know how each of the elements in the N x C matrix is
    # contributing to the loss
    # Intuitively, as most elements in the non-sparse cross entropy will be zero,
    # (as it will be multiplied with 0 probability)
    # these will not contribute to the loss
    # Sticking to the PyTorch nomenclature, call the ground truth y and the
    # prediction x, we want to take the derivative w.r.t the predictions x
    # There will be one prediction per element, i.e. x = x(n,c) where n is a
    # specific batch and c a specific class
    # To make the calculation simple for ourselves we've chopped the expression,
    # so that we don't need to take the derivative of x(n,c) directly
    # Instead, we will take the derivative w.r.t to the immediate variable
    # logprobs(x(n,c))
    # I.e. for each element we will take the derivative dl/d(logprobs(x(n,c)))
    # The loss function with the reduction can be written as
    # - 1/N sum_N sum_C y_nc * logprobs(x(n,c))
    # Most y_nc's will be zeros, the rest will be ones, hence the
    # "surviving terms" can be written as
    # d/d(logprobs(x(n,c))) (- 1/N 1 * logprobs(x(n,c)) = - 1/N
    dl_d_log_probabilities = torch.zeros_like(log_probabilities)
    batch_size = embedding.size(dim=0)
    dl_d_log_probabilities[range(batch_size), targets] = -(1.0 / batch_size)
    # dl/d(probs) = dl/d(log(probs)) * d(log(probs))/d(probs)
    # From above, we calculated dl/d(log(probs)), and
    # d log(x)/dx = (1/x)
    dl_d_probabilities = (1.0 / probabilities) * dl_d_log_probabilities
    # dl/d(counts_sum_inv) = dl/d(probs) * d(probs)/d(counts_sum_inv)
    # We have dl/d(probs) from above
    # Further, we have that
    # probs = counts * counts_sum_inv
    # so
    # d probs/ d counts_sum_inv = counts
    # However, counts has dimension (N,C) and counts_sum_inv has dimension (N, 1)
    # because we summed over the C dimension in the counts_sum variable
    # Broadcasting rules
    # https://pytorch.org/docs/stable/notes/broadcasting.html
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    # tells us that count_sums_inv dimension 1 will be broadcasted
    # With an example, lets probs : R^{2x2} and counts_sum_inv : R^{2x1}
    # probs =
    # [[a_00, a_01],
    #  [a_10, a_11]]
    # counts_sum_inv =
    # [[b_00],
    #  [b_10]]
    # counts_sum_inv will be stretched in the direction of dimension 1 and become
    # counts_sum_inv =
    # [[b_00, b_00],
    #  [b_10, b_10]]
    # as the multiplication is element-wise, we get
    # probs*counts_sum_inv =
    # [[a_00*b_00, a_01*b_00],
    #  [a_10*b_10, a_11*b_10]]
    # Since counts_sum_inv is replicated we must accumulate the gradient
    # (recall from micrograd when the same value was an input to several other
    #  nodes)
    # We will therefore sum the gradient over the columns
    dl_d_counts_sum_inv = (counts * dl_d_probabilities).sum(dim=1, keepdim=True)
    # counts appears twice
    # 1. probabilities = counts * counts_sum_inv
    # 2. counts_sum = counts.sum(1, keepsdim=True)
    # We must therefore accumulate the gradients
    # 1. dl/d(counts) = dl/d(probabilities) * d(probabilities)/d(counts)
    #    dl/d(counts) = dl/d(probabilities) * counts_sum_inv
    #    counts_sum_inv has dimension (N,1)
    #    dl/d(probabilities) has dimension (N,C)
    #    counts_sum_inv will therefore be stretched in the C dimension
    # 2. dl/d(counts) = dl/d(counts_sum) * d(counts_sum)/d(counts)
    # However, we don't know dl/d(counts_sum yet, but it can be calculated
    # Do let's calculate that first
    # 3. dl/d(counts_sum) = dl/d(counts_sum_inv) * d(counts_sum_inv)/d(counts_sum)
    #    Since d/dx (1/x) = -1/(x^2), we get
    #    dl/d(counts_sum) = dl/d(counts_sum_inv) * (-1/counts^2)
    dl_d_counts_sum = torch.zeros_like(counts_sum)
    dl_d_counts = torch.zeros_like(counts)
    dl_d_normalized_logits = torch.zeros_like(normalized_logits)
    dl_d_logits_maxes = torch.zeros_like(logits_maxes)
    dl_d_logits = torch.zeros_like(logits)
    # Calculate the derivatives of the second layer
    dl_d_h = torch.zeros_like(h)
    dl_d_h_pre_activation = torch.zeros_like(h_pre_activation)
    dl_d_w2 = torch.zeros_like(w2)
    dl_d_b2 = torch.zeros_like(b2)
    # Calculate the derivatives of the batch norm layer (of the first layer)
    dl_d_batch_normalization_raw = torch.zeros_like(batch_normalization_raw)
    dl_d_inv_batch_normalization_std = torch.zeros_like(inv_batch_normalization_std)
    dl_d_batch_normalization_var = torch.zeros_like(batch_normalization_var)
    dl_d_batch_normalization_diff_squared = torch.zeros_like(
        batch_normalization_diff_squared
    )
    dl_d_batch_normalization_diff = torch.zeros_like(batch_normalization_diff)
    dl_d_batch_normalization_mean = torch.zeros_like(batch_normalization_mean)
    dl_d_h_pre_batch_norm = torch.zeros_like(h_pre_batch_norm)
    dl_d_batch_normalization_gain = torch.zeros_like(batch_normalization_gain)
    dl_d_batch_normalization_bias = torch.zeros_like(batch_normalization_bias)
    # Calculate the derivatives of the first layer
    dl_d_w1 = torch.zeros_like(w1)
    dl_d_b1 = torch.zeros_like(b1)
    # Calculate the derivatives of the embedding layer
    dl_d_concatenated_embedding = torch.zeros_like(concatenated_embedding)
    dl_d_embedding = torch.zeros_like(embedding)
    dl_d_c = torch.zeros_like(c)

    gradients: Dict[str, torch.Tensor] = {}
    gradients["dl_d_log_probabilities"] = dl_d_log_probabilities
    gradients["dl_d_probabilities"] = dl_d_probabilities
    gradients["dl_d_counts_sum_inv"] = dl_d_counts_sum_inv
    gradients["dl_d_counts_sum"] = dl_d_counts_sum
    gradients["dl_d_counts"] = dl_d_counts
    gradients["dl_d_normalized_logits"] = dl_d_normalized_logits
    gradients["dl_d_logits_maxes"] = dl_d_logits_maxes
    gradients["dl_d_logits"] = dl_d_logits
    gradients["dl_d_h"] = dl_d_h
    gradients["dl_d_h_pre_activation"] = dl_d_h_pre_activation
    gradients["dl_d_w2"] = dl_d_w2
    gradients["dl_d_b2"] = dl_d_b2
    gradients["dl_d_batch_normalization_raw"] = dl_d_batch_normalization_raw
    gradients["dl_d_inv_batch_normalization_std"] = dl_d_inv_batch_normalization_std
    gradients["dl_d_batch_normalization_var"] = dl_d_batch_normalization_var
    gradients["dl_d_batch_normalization_diff_squared"] = (
        dl_d_batch_normalization_diff_squared
    )
    gradients["dl_d_batch_normalization_diff"] = dl_d_batch_normalization_diff
    gradients["dl_d_batch_normalization_mean"] = dl_d_batch_normalization_mean
    gradients["dl_d_h_pre_batch_norm"] = dl_d_h_pre_batch_norm
    gradients["dl_d_batch_normalization_gain"] = dl_d_batch_normalization_gain
    gradients["dl_d_batch_normalization_bias"] = dl_d_batch_normalization_bias
    gradients["dl_d_w1"] = dl_d_w1
    gradients["dl_d_b1"] = dl_d_b1
    gradients["dl_d_concatenated_embedding"] = dl_d_concatenated_embedding
    gradients["dl_d_embedding"] = dl_d_embedding
    gradients["dl_d_c"] = dl_d_c

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
    w2.grad = gradients["dl_d_w2"]
    b2.grad = gradients["dl_d_b2"]
    # Gradients of the batch norm layer
    batch_normalization_gain.grad = gradients["dl_d_batch_normalization_gain"]
    batch_normalization_bias.grad = gradients["dl_d_batch_normalization_bias"]
    # Gradients of the first layer
    w1.grad = gradients["dl_d_w1"]
    b1.grad = gradients["dl_d_b1"]
    # Gradients of the embedding layer
    c.grad = gradients["dl_d_c"]


def compare_gradients(
    model: torch.Tensor,
    intermediate_variables: Dict[str, torch.Tensor],
    gradients: Dict[str, torch.Tensor],
):
    """
    Compare the manually calculated gradients with the ones generated from autograd.

    Raises:
        RuntimeError: In case not all the tensors are approximately equal

    Args:
        model (torch.Tensor): The model weights
        intermediate_variables (Dict[str, torch.Tensor]): The intermediate
            variables
        gradients (Dict[str, torch.Tensor]): The manually calculated gradients
    """
    approximate_bool_list = []
    # Make a model dict for easier comparison
    model_dict: Dict[str, torch.Tensor] = {}
    (
        model_dict["c"],
        model_dict["w1"],
        model_dict["b1"],
        model_dict["w2"],
        model_dict["b2"],
        model_dict["batch_normalization_gain"],
        model_dict["batch_normalization_bias"],
    ) = model
    print("Comparing model weights:")
    print("-" * 80)
    for variable_name, tensor in model_dict.items():
        approximate_bool_list.append(
            compare_single_gradient(
                name=variable_name,
                manually_calculated=gradients[f"dl_d_{variable_name}"],
                tensor=tensor,
            )
        )

    print("\nComparing intermediate variables:")
    print("-" * 80)
    for variable_name in intermediate_variables.keys():
        approximate_bool_list.append(
            compare_single_gradient(
                name=variable_name,
                manually_calculated=gradients[f"dl_d_{variable_name}"],
                tensor=intermediate_variables[variable_name],
            )
        )

    if not all(approximate_bool_list):
        raise RuntimeError("Some of the gradients are off, see output above for debug")


def compare_single_gradient(
    name: str, manually_calculated: torch.Tensor, tensor: torch.Tensor
) -> bool:
    """
    Compare the manually calculated gradient with the one calculated using autograd.

    Args:
        name (str): Name of the tensor
        manually_calculated (torch.Tensor): The manually calculated gradient
        tensor (torch.Tensor): The tensor to check

    Returns:
        bool: Whether the tensor is approximately equal
    """
    exact = torch.all(manually_calculated == tensor.grad).item()
    approximate = torch.allclose(manually_calculated, tensor.grad)
    max_diff = (manually_calculated - tensor.grad).abs().max().item()
    print(
        f"{name:32s} | "
        f"exact: {str(exact):5s} | "
        f"approximate {str(approximate):5s} | "
        f"max difference: {max_diff}"
    )

    return approximate


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
