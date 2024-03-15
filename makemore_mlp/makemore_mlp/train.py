"""Module to train the model."""

from typing import Tuple

import torch

def train_neural_net_model(
    model: Tuple[torch.Tensor, ...],
    input_data: torch.Tensor,
    ground_truth_data: torch.Tensor,
    n_mini_batches: int,
    batch_size: int
) -> Tuple[torch.Tensor]:
    """Train the neural net model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model (weights) to use
        input_data (torch.Tensor): The input training data (the data fed into the
            features)
        ground_truth_data (torch.Tensor): The correct prediction for the input 
            (the correct labels)
        n_mini_batches (int): Number of mini batches to use
        batch_size (int): The batch size to use

    Returns:
        torch.Tensor: The trained model
    """
    # Alias
    c, w1, b1, w2, b2 = model

    for _ in range(n_mini_batches):
        # Mini batch constructor
        n_samples = input_data.shape[0]
        idxs = torch.randint(low=0, high=n_samples, size=batch_size)

        # NOTE: input_data has dimension (n_samples, block_size)
        #       input_data[idxs] selects batch_size samples from the training 
        #       data
        #       The size of input_data[idxs] is therefore 
        #       (batch_size, block_size)
        #       c has dimension (VOCAB_SIZE, embedding_size)
        #       c[input_data[idxs]] will grab embedding_size vectors
        #       for each of the block_size characters
        #       The dimension of emb is therefore 
        #       (batch_size, block_size, embedding_size)
        emb = c[input_data[idxs]]
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
        # The logits will have dimension (batch_size, VOCAB_SIZE)
        logits = h@w2 + b2
        # Get the fake counts (like in makemore_bigram)
        counts = logits.exp()
        # Normalize to probability
        prob = counts/counts.sum(1, keepdim=True)

        # Negative loss likelihood:
        # - For each of the example in the batch: torch.arange(batch_size)
        #   - Select the probability of getting the ground truth:
        #     prob[torch.arange(batch_size), ground_truth_data]
        #     The shape of this will be the batch_size
        #     In a model that perfectly predicts the characters all entries in
        #     prob[torch.arange(batch_size), ground_truth_data]
        #     would be 1
        # - We take the logarithm of this, so that numbers under 1 becomes
        #   negative (in perfect prediction log(1)=0)
        #   - In order to get a positive number to minimize we take the negative
        #     of the result
        # - Finally, the mean of this the number we want to minimize
        loss = -prob[torch.arange(batch_size), ground_truth_data].log().mean()

    return c, w1, b1, w2, b2