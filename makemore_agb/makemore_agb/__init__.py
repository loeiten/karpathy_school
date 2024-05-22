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

# NOTE: In this case the CPU might in fact be the fastest due to the following
#       reasons
#       1. The model is small, which might require a lot of memory movement
#       2. No I/O overlap is implemented
#       3. I've (most likely) done something really silly
#
#       Alternatively one could've made the device like this
#       DEVICE = (
#           torch.device("cuda")
#           if torch.cuda.is_available()
#           else (
#               torch.device("mps")
#               if torch.backends.mps.is_available()
#               else torch.device("cpu")
#           )
#       )
DEVICE = torch.device("cpu")
