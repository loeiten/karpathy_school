"""Module to run inference on the model."""

from typing import Tuple

import torch


def predict_neural_network(
    model: Tuple[torch.Tensor, ...],
    input_data: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """Predict the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_data (torch.Tensor): The data to run inference on.
            This data has the shape (batch_size, block_size)

    Returns:
        torch.Tensor: The achieved logits with shape (batch_size)
    """
    # Alias
    c, w1, b1, w2, b2 = model
    # NOTE: c has dimension (VOCAB_SIZE, embedding_size)
    #       input_data has the dimension (batch_size, block_size)
    #       c[input_data] will grab embedding_size vectors for each of the
    #       block_size characters
    #       The dimension of emb is therefore
    #       (batch_size, block_size, embedding_size)
    emb = c[input_data]
    # NOTE: Given a block_size of 3 and an embedding size of 2, we could have
    #       done the following:
    #
    #       emb = C[X]
    #       # The first dimension of C[X] would be the number of parameters
    #       # The second would be the number of block_size
    #       # The last dimension would be the embedding_size
    #       torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]])
    #
    #       However, this would fix the code to use block_size = 2
    #       as 0 will be the first character in the block, 1 will be the
    #       second and so on
    #
    #       Another approach could be to use
    #       torch.cat(torch.unbind(emb, 1), 1)
    #       Where torch.unbind which splits the tensor
    #       to a tuple of tensors along the desired dimension
    #
    #       However, this would create a new tensor
    #       Instead, we can just change it's view
    #       emb.view(n_samples, block_size*embedding_size)
    # The block needs to be concatenated before multiplying it with the
    # weight
    # That is, the dimension size will be block_size*embedding_size
    concatenated_dimension_size = emb.shape[1] * emb.shape[2]
    # NOTE: .view(-1, x) - the -1 will make pyTorch infer the dimension for
    #       that dimension
    # NOTE: + b1 is broadcasting on the correct dimension
    # NOTE: The broadcasting will succeed
    h = torch.tanh(emb.view(-1, concatenated_dimension_size) @ w1 + b1)
    # The logits will have dimension (batch_size, VOCAB_SIZE)
    logits = h @ w2 + b2

    return logits
