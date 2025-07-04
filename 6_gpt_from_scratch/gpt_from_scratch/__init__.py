"""Package containing gpt_from_scratch."""

import torch

torch.manual_seed(192837465)  # type: ignore

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)

print(f"Running on {DEVICE}")
