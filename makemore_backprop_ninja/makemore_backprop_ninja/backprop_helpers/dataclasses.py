"""File containing data classes."""

from enum import Enum


class BackpropMode(Enum):
    """
    Contains backprop modes.

    AUTOMATIC - Uses pytorch autograd
    VERBOSE - Calculate the gradients by hand in an verbose way
    """

    AUTOMATIC = "AUTOMATIC"
    VERBOSE = "VERBOSE"
