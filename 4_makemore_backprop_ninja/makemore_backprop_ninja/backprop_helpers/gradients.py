"""Module dealing with gradients."""

from typing import Dict, Tuple

import torch


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
    approximate = torch.allclose(manually_calculated, tensor.grad, atol=1.2e-8)
    max_diff = (manually_calculated - tensor.grad).abs().max().item()
    print(
        f"{name:32s} | "
        f"exact: {str(exact):5s} | "
        f"approximate {str(approximate):5s} | "
        f"max difference: {max_diff}"
    )

    return approximate


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
