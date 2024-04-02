"""Module containing option classes."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainStatistics:
    """Class holding train statistics."""

    train_loss: List[float] = field(default_factory=list)
    train_step: List[int] = field(default_factory=list)
    eval_loss: List[float] = field(default_factory=list)
    eval_step: List[int] = field(default_factory=list)


@dataclass
class ModelOptions:
    """Class holding possible model option."""

    n_mini_batches: int = 10000
    batch_size: int = 32
    # NOTE: How to find the optimal learning rate (roughly)
    #       1. Find the range where in the low end the loss barely moves and
    #          where it explodes
    #       2. Make a linspace in the exponent
    #       3. Make one training step with the exponent
    #       4. Plot and see where have a minima
    #       One can then run with the optimal learning rate for a while and
    #       gradually decay it
    learning_rate: float = 0.1
