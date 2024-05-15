"""Module to run inference on the model."""

from typing import Tuple

import torch


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable=too-many-locals
def predict_neural_network(
    model: Tuple[torch.Tensor, ...],
    input_data: torch.Tensor,
    inspect_pre_activation_and_h: bool = False,
    batch_normalize: bool = False,
    training: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Predict the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_data (torch.Tensor): The data to run inference on.
            This data has the shape (batch_size, block_size)
        inspect_pre_activation_and_h (bool): Whether or not to output the
            pre-activation and activation
        batch_normalize (bool): Whether or not to batch normalize
        training (bool): Flag to keep track of whether we're training or not

    Returns:
        torch.Tensor: The achieved logits with shape (batch_size)
    """
    # Alias
    (
        c,
        w1,
        b1,
        w2,
        b2,
        batch_normalization_gain,
        batch_normalization_bias,
        batch_normalization_mean_running,
        batch_normalization_std_running,
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
    h_pre_activation = (concatenated_embedding @ w1) + b1
    if batch_normalize:
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
                # Add small updates
                batch_normalization_mean_running = (
                    0.999 * batch_normalization_mean_running
                    + 0.001 * batch_normalization_mean
                )
                batch_normalization_std_running = (
                    0.999 * batch_normalization_std_running
                    + 0.001 * batch_normalization_std
                )
        else:
            batch_normalization_mean = batch_normalization_mean_running
            batch_normalization_std = batch_normalization_std_running

        h_pre_activation = (
            batch_normalization_gain
            * (h_pre_activation - batch_normalization_mean)
            / batch_normalization_std
        ) + batch_normalization_bias

    h = torch.tanh(h_pre_activation)
    # The logits will have dimension (batch_size, VOCAB_SIZE)
    logits = h @ w2 + b2

    if not inspect_pre_activation_and_h:
        return (logits,)
    return (logits, h_pre_activation, h)
