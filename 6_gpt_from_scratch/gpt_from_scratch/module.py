"""Module for the abstract module class."""

import abc
from typing import List, Tuple

import torch


class Module(metaclass=abc.ABCMeta):
    """
    Class mimicking the torch.nn.Module class in Pytorch.

    NOTE: This is not how Pytorch abstracts this class
    We implement this here only to improve the typing elsewhere.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the layer.

        NOTE: self.out is not in the real PyTorch Module, but we use it to
              inspect the output in different experiments
        """
        self.out: torch.Tensor

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """

    @abc.abstractmethod
    def parameters(self) -> Tuple[torch.Tensor, ...]:
        """Return the parameters.

        Returns:
            Tuple[torch.Tensor,...]: Tuple of the parameters
        """


class Sequential(Module):
    """Class mimicking torch.nn.Sequential."""

    def __init__(self, layers: List[Module]) -> None:
        """Set the layers.

        Args:
            layers (List[Module]): Layers to be added to the sequence
        """
        self.layers = layers
        # NOTE: Just for us in order to inspect the output
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run the sequence on input data.

        Args:
            x (torch.Tensor): The input data

        Returns:
            torch.Tensor: The result of passing the data through the layers
        """
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self) -> Tuple[torch.Tensor, ...]:
        """Return the parameters of the Sequence.

        Returns:
            Tuple[torch.Tensor,...]: Tuple of the parameters
        """
        return tuple(
            parameter for layer in self.layers for parameter in layer.parameters()
        )
