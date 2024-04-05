"""Module containing option classes."""

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class TrainStatistics:
    """Class holding train statistics."""

    train_loss: List[float] = field(default_factory=list)
    train_step: List[int] = field(default_factory=list)
    eval_loss: List[float] = field(default_factory=list)
    eval_step: List[int] = field(default_factory=list)


@dataclass
class OptimizationParams:
    """Class holding possible optimization option."""

    total_mini_batches: int = 10_000
    mini_batches_per_iteration: int = 100
    batch_size: int = 32
    n_iterations: int = total_mini_batches // mini_batches_per_iteration
    cur_mini_batch: int = 0
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
    """Class holding possible model options."""

    block_size: int = 3
    embedding_size: int = 2
    hidden_layer_neurons: int = 100