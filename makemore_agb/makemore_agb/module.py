"""Module for the abstract module class."""

import abc
from typing import List

import torch


class Module(metaclass=abc.ABCMeta):
    """
    Class mimicking the torch.nn.Module class in Pytorch.

    NOTE: This is not how Pytorch abstracts this class
    We implement this here only to improve the typing elsewhere.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the layer."""
        self.weight: torch.Tensor

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """

    @abc.abstractmethod
    def parameters(self) -> List[torch.Tensor]:
        """Return the parameters.

        Returns:
            List[torch.Tensor]: The parameters of the layer
        """
