"""Module to train the model."""

from typing import Tuple

import torch

def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    training_data: torch.Tensor,
    n_mini_batches: int,
    batch_size: int
) -> Tuple[torch.Tensor]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        training_data (torch.Tensor): The training data
        n_mini_batches (int): Number of mini batches to use
        batch_size (int): The batch size to use

    Returns:
        torch.Tensor: The trained model
    """
    # Alias
    c, w1, b1, w2, b2 = model

    for _ in range(n_mini_batches):
        # Mini batch constructor
        n_samples = training_data.shape[0]
        idxs = torch.randint(low=0, high=n_samples, size=batch_size)

        # NOTE: training_data has dimension (n_samples, block_size)
        #       training_data[idxs] selects batch_size samples from the training 
        #       data
        #       The size of training_data[idxs] is therefore 
        #       (batch_size, block_size)
        #       c has dimension (VOCAB_SIZE, embedding_size)
        #       c[training_data[idxs]] will grab embedding_size vectors
        #       for each of the block_size characters
        #       The dimension of emb is therefore 
        #       (batch_size, block_size, embedding_size)
        emb = c[training_data[idxs]]
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
        concatenated_dimension_size = emb.shape[1]*emb.shape[2]
        # NOTE: .view(-1, x) - the -1 will make pyTorch infer the dimension for 
        #       that dimension
        # NOTE: + b1 is broadcasting on the correct dimension
        # NOTE: The broadcasting will succeed
        h = torch.tanh(emb.view(-1, concatenated_dimension_size) @ w1 + b1)

    return c, w1, b1, w2, b2