"""Module to run inference on the model."""

from typing import List, Tuple

import torch

from makemore_mlp import INDEX_TO_TOKEN, TOKEN_TO_INDEX


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


def run_inference(
    model: Tuple[torch.Tensor, ...],
    n_samples: int = 20,
    seed: int = 2147483647,
) -> Tuple[str, ...]:
    """Run inference on the model.

    Args:
        model (Tuple[torch.Tensor, ...]): The model to run inference on.
        n_samples (int, optional): The number of inferences to run.
            Defaults to 20.
        seed (int, optional): The seed to use. Defaults to 2147483647.

    Returns:
        Tuple[str, ...]: The predictions
    """
    # Obtain the embedding size from c
    embedding_size = int(model[0].shape[-1].item())
    # Obtain the block size from w1
    block_size = int(model[1].shape[-2].item() / embedding_size)

    g = torch.Generator().manual_seed(seed)
    predictions: List[str] = []

    for _ in range(n_samples):
        characters = ""
        context = [TOKEN_TO_INDEX["."]] * block_size  # Initialize with stop characters

        while True:
            logits = predict_neural_network(
                model=model, input_data=torch.tensor(context)
            )
            probs = torch.softmax(logits, dim=1)
            index = torch.multinomial(probs, num_samples=1, generator=g)
            # The context size is constant, so we drop the first token, and add
            # the predicted token to the next
            context = context[1:] + [index]
            characters += f"{INDEX_TO_TOKEN[index]}"
            if index == 0:
                break

    return tuple(predictions)
