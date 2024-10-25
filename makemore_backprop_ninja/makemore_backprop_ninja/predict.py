"""Module to run inference on the model."""

from typing import Tuple

import torch
from makemore_backprop_ninja.data_classes import BatchNormalizationParameters


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-locals
def predict_neural_network(
    model: Tuple[torch.Tensor, ...],
    batch_normalization_parameters: BatchNormalizationParameters,
    input_data: torch.Tensor,
    training: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Predict using the explicit network model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        batch_normalization_parameters (Optional[BatchNormalizationParameters]):
            Contains the running mean and the running standard deviation
        input_data (torch.Tensor): The data to run inference on.
            This data has the shape (batch_size, block_size)
        training (bool): Flag to keep track of whether we're training or not

    Returns:
        torch.Tensor: The achieved logits with shape (batch_size)
    """
    # Alias
    if btch_normalization_parameters is not None:
        (
            c,
            w1,
            b1,
            w2,
            b2,
            batch_normalization_gain,
            batch_normalization_bias,
        ) = model
    else:
        (
            c,
            w1,
            b1,
            w2,
            b2,
        ) = model
    # NOTE: c has dimension (VOCAB_SIZE, embedding_size)
    #       input_data has the dimension (batch_size, block_size)
    #       c[input_data] will grab embedding_size vectors for each of the
    #       block_size characters
    #       The dimension of emb is therefore
    #       (batch_size, block_size, embedding_size)
    embedding = c[input_data]
    # The block needs to be concatenated before multiplying it with the
    # weight
    # That is, the dimension size will be block_size*embedding_size
    # Another way to look at it is that we need the batch size to stay the
    # same, whereas the second dimension should be the rest should be squashed
    # together
    concatenated_embedding = embedding.view(embedding.shape[0], -1)
    # NOTE: + b1 is broadcasting on the correct dimension
    # NOTE: When we are using batch normalization b1 will get cancelled out by
    #       subtracting batch_normalization_mean, and the gradient will become
    #       zero.
    #       batch_normalization_bias will in any case play the role as bias in
    #       this pre activation layer.
    #       It's therefore wasteful to use this add operation when we're using
    #       batch normalization
    h_pre_activation = (concatenated_embedding @ w1) + b1

    if training:
        # Note that batch normalization couples the batch together
        # That is: The activation is no longer a function of the example itself,
        # but also what batch it arrived with
        # It turns out that this adds some entropy to the system which works as
        # a regularizer, and makes it harder for the model to overfit
        # However, when we are doing inference, what mean and std should we use?
        # One could take the mean and std over the whole data set as a final
        # step during the training, but having a running updates in the
        # direction of the current mean and stddev
        batch_normalization_mean = h_pre_activation.mean(0, keepdim=True)
        batch_normalization_std = h_pre_activation.std(0, keepdim=True)

        with torch.no_grad():
            # Here we use a momentum of 0.001
            # We expect that for large batches the mean and std ar going to
            # be roughly the same
            # However, here we use small batch sizes and the values may
            # fluctuate
            # Using a lower momentum ensures that we do not overshoot
            batch_normalization_parameters.running_mean = (
                0.999 * batch_normalization_parameters.running_mean
                + 0.001 * batch_normalization_mean
            )
            batch_normalization_parameters.running_std = (
                0.999 * batch_normalization_parameters.running_std
                + 0.001 * batch_normalization_std
            )
    else:
        batch_normalization_mean = batch_normalization_parameters.running_mean
        batch_normalization_std = batch_normalization_parameters.running_std

    h_pre_activation = (
        batch_normalization_gain
        * (h_pre_activation - batch_normalization_mean)
        / batch_normalization_std
    ) + batch_normalization_bias

    h = torch.tanh(h_pre_activation)
    # The logits will have dimension (batch_size, VOCAB_SIZE)
    logits = h @ w2 + b2

    return logits
