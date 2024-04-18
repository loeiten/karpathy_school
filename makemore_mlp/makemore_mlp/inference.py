"""Module to run inference on the model."""

from typing import List, Tuple

import torch
from makemore_mlp.predict import predict_neural_network

from makemore_mlp import INDEX_TO_TOKEN, TOKEN_TO_INDEX


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
    embedding_size = int(model[0].shape[-1])
    # Obtain the block size from w1
    block_size = int(model[1].shape[-2] / embedding_size)

    g = torch.Generator().manual_seed(seed)
    predictions: List[str] = []

    for _ in range(n_samples):
        characters = ""
        context = [TOKEN_TO_INDEX["."]] * block_size  # Initialize with stop characters

        while True:
            # Note the [] to get the batch shape correct
            logits = predict_neural_network(
                model=model, input_data=torch.tensor([context])
            )
            probs = torch.softmax(logits, dim=1)
            index = torch.multinomial(probs, num_samples=1, generator=g)
            # The context size is constant, so we drop the first token, and add
            # the predicted token to the next
            context = context[1:] + [index]
            characters += f"{INDEX_TO_TOKEN[index.item()]}"
            if index == 0:
                break

        predictions.append(characters)

    return tuple(predictions)
