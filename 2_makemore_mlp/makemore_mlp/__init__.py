"""Package containing makemore_mlp."""

# +1 as we'd like the start and stop token to have value 0
TOKEN_TO_INDEX = {chr(ord("a") + i): i + 1 for i in range(26)}
# Start/stop token
TOKEN_TO_INDEX["."] = 0
INDEX_TO_TOKEN = {value: key for key, value in TOKEN_TO_INDEX.items()}

VOCAB_SIZE = len(TOKEN_TO_INDEX.keys())
