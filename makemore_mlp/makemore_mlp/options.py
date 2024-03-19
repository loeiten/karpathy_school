"""Module containing option classes."""

from dataclasses import dataclass


@dataclass
class ModelOptions:
    """Class holding possible model option."""

    n_mini_batches: int = 10000
    batch_size: int = 32
    learning_rate: float = -0.1
