"""File containing data classes."""

from enum import Enum


class BackpropMode(Enum):
    """
    Contains backprop modes.

    AUTOMATIC - Uses pytorch autograd
    VERBOSE - Calculate the gradients by hand in a verbose way
    SUCCINCT - Calculate the gradients by hand in a succinct way
    """

    AUTOMATIC = "AUTOMATIC"
    VERBOSE = "VERBOSE"
    SUCCINCT = "SUCCINCT"
