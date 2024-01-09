"""Package containing makemore_bigram."""

ALPHABET_DICT = {chr(ord("a") + i): i for i in range(26)}
ALPHABET_DICT["."] = 26
INVERSE_ALPHABET_DICT = {value: key for key, value in ALPHABET_DICT.items()}
