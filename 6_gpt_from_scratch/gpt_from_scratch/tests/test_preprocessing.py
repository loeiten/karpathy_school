"""Contains tests for the preprocessing module."""

from gpt_from_scratch.preprocessing import CharTokenizer


def test_CharTokenizer__init__() -> None:
    """Test the initializer of CharTokenizer."""
    char_tokenizer = CharTokenizer("qwerty")
    assert char_tokenizer.tokens == ["e", "q", "r", "t", "w", "y"]
    assert char_tokenizer.vocab_size == 6
