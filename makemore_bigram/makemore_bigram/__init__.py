"""Package containing makemore_bigram."""

# +1 as we'd like the start and stop token to have value 0
ALPHABET_DICT = {chr(ord("a") + i): i + 1 for i in range(26)}
# Start/stop token
ALPHABET_DICT["."] = 0
INVERSE_ALPHABET_DICT = {value: key for key, value in ALPHABET_DICT.items()}
