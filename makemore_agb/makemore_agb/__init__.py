"""Package containing makemore_agb."""

from typing import Dict, Literal

import torch

# +1 as we'd like the start and stop token to have value 0
TOKEN_TO_INDEX = {chr(ord("a") + i): i + 1 for i in range(26)}
# Start/stop token
TOKEN_TO_INDEX["."] = 0
INDEX_TO_TOKEN = {value: key for key, value in TOKEN_TO_INDEX.items()}

VOCAB_SIZE = len(TOKEN_TO_INDEX.keys())

DATASET = Dict[
    Literal[
        "training_input_data",
        "training_ground_truth",
        "validation_input_data",
        "validation_ground_truth",
        "test_input_data",
        "test_ground_truth",
    ],
    torch.Tensor,
]
