"""Module containing option classes."""

from dataclasses import dataclass, field
from typing import Callable, List
from enum import Enum


@dataclass
class TrainStatistics:
    """Class holding train statistics."""

    training_loss: List[float] = field(default_factory=list)
    training_step: List[int] = field(default_factory=list)
    eval_training_loss: List[float] = field(default_factory=list)
    eval_training_step: List[int] = field(default_factory=list)
    eval_validation_loss: List[float] = field(default_factory=list)
    eval_validation_step: List[int] = field(default_factory=list)


@dataclass
class OptimizationParams:
    """Class holding possible optimization option."""

    n_mini_batches: int = 200_000
    mini_batches_per_data_capture: int = 1_000
    batch_size: int = 32
    cur_step: int = 0
    # NOTE: How to find the optimal learning rate (roughly)
    #       1. Find the range where in the low end the loss barely moves and
    #          where it explodes
    #       2. Make a linspace in the exponent
    #       3. Make one training step with the exponent
    #       4. Plot and see where have a minima
    #       One can then run with the optimal learning rate for a while and
    #       gradually decay it
    learning_rate: Callable[[int], float] = lambda cur_mini_batch: (
        0.1 if cur_mini_batch < 100_000 else 0.01
    )


@dataclass
class ModelParams:
    """
    Class holding possible model options.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length
        embedding_size (int): The size of the embedding
        hidden_layer_neurons (int): The seed for the random number generator
        seed (int): The seed for the random number generator
    """

    block_size: int = 3
    embedding_size: int = 2
    hidden_layer_neurons: int = 100
    seed: int = 2147483647


class ModelType(Enum):
    """
    Contains the model types.

    NONE - Do not use pre-canned ModelType
    ORIGINAL_12K - The original 12k parameter model
    """

    NONE = "NONE"
    ORIGINAL_12K = "ORIGINAL_12K"
    CONTEXT_8_22K = "CONTEXT_8_22K"