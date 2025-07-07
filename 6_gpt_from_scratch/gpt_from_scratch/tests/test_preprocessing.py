"""Contains tests for the preprocessing module."""

import pytest
from gpt_from_scratch.preprocessing import CharTokenizer


@pytest.fixture
def qwerty_tokenizer() -> CharTokenizer:
    """Return the qweErTy tokenizer.

    Returns:
        CharTokenizer: The tokenizer
    """
    return CharTokenizer("qweErTy")


def test_CharTokenizer__init__(qwerty_tokenizer: CharTokenizer) -> None:
    """Test the initializer of CharTokenizer.

    Args:
        qwerty_tokenizer (CharTokenizer): The qweErTy tokenizer
    """
    # NOTE: sort sorts according to their ASCII values
    assert qwerty_tokenizer.tokens == ["E", "T", "e", "q", "r", "w", "y"]
    assert qwerty_tokenizer.vocab_size == 7


def test_CharTokenizer__encode__(qwerty_tokenizer: CharTokenizer) -> None:
    """Test the CharTokenizer encoder.

    Args:
        qwerty_tokenizer (CharTokenizer): The qweErTy tokenizer
    """
    assert qwerty_tokenizer.encode("ETeqrwy") == [0, 1, 2, 3, 4, 5, 6]

    with pytest.raises(KeyError):
        qwerty_tokenizer.encode("a")

    assert qwerty_tokenizer.encode("weTTer") == [5, 2, 1, 1, 2, 4]


def test_CharTokenizer__decode__(qwerty_tokenizer: CharTokenizer) -> None:
    """Test the CharTokenizer decoder.

    Args:
        qwerty_tokenizer (CharTokenizer): The qweErTy tokenizer
    """
    assert qwerty_tokenizer.decode([0, 1, 2, 3, 4, 5, 6]) == "ETeqrwy"

    with pytest.raises(KeyError):
        qwerty_tokenizer.decode([7])

    assert qwerty_tokenizer.decode([5, 2, 1, 1, 2, 4]) == "weTTer"
