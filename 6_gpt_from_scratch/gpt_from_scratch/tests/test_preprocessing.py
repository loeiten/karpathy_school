"""Contains tests for the preprocessing module."""

import torch
import pytest
from gpt_from_scratch.preprocessing import CharTokenizer, DataPreprocessor, DataSet


@pytest.fixture(scope="session")
def qwerty_tokenizer() -> CharTokenizer:
    """Return the qweErTy tokenizer.

    Returns:
        CharTokenizer: The tokenizer
    """
    return CharTokenizer("qweErTy")


def test_CharTokenizer___init__(qwerty_tokenizer: CharTokenizer) -> None:
    """Test the initializer of CharTokenizer.

    Args:
        qwerty_tokenizer (CharTokenizer): The qweErTy tokenizer
    """
    # NOTE: sort sorts according to their ASCII values
    assert qwerty_tokenizer.tokens == ["E", "T", "e", "q", "r", "w", "y"]
    assert qwerty_tokenizer.vocab_size == 7


def test_CharTokenizer_encode(qwerty_tokenizer: CharTokenizer) -> None:
    """Test the CharTokenizer encoder.

    Args:
        qwerty_tokenizer (CharTokenizer): The qweErTy tokenizer
    """
    assert qwerty_tokenizer.encode("ETeqrwy") == [0, 1, 2, 3, 4, 5, 6]

    with pytest.raises(KeyError):
        qwerty_tokenizer.encode("a")

    assert qwerty_tokenizer.encode("weTTer") == [5, 2, 1, 1, 2, 4]


def test_CharTokenizer_decode(qwerty_tokenizer: CharTokenizer) -> None:
    """Test the CharTokenizer decoder.

    Args:
        qwerty_tokenizer (CharTokenizer): The qweErTy tokenizer
    """
    assert qwerty_tokenizer.decode([0, 1, 2, 3, 4, 5, 6]) == "ETeqrwy"

    with pytest.raises(KeyError):
        qwerty_tokenizer.decode([7])

    assert qwerty_tokenizer.decode([5, 2, 1, 1, 2, 4]) == "weTTer"


def test_DataPreprocessor___init__() -> None:
    """Test the initializer of DataPreprocessor."""
    # NOTE: The init is pointing to the tiny Shakespare corpus
    context_length = 3
    data_preprocessor = DataPreprocessor(context_length=context_length)
    assert data_preprocessor.train_split.size() == torch.Size([1003851])
    assert data_preprocessor.validation_split.size() == torch.Size([111539])
    assert data_preprocessor.context_length == context_length


def test_DataPreprocessor_get_batch() -> None:
    """Test the get batch of DataPreprocessor."""
    context_length = 3
    batch_size = 4
    dp = DataPreprocessor(context_length=context_length)
    xb, yb = dp.get_batch(sample_from=DataSet.TRAIN, batch_size=batch_size)
    assert xb.size() == torch.Size([batch_size, context_length])
    assert yb.size() == torch.Size([batch_size, context_length])
    # Loop through the batches
    for cur_batch_x, cur_batch_y in zip(xb, yb):
        assert torch.equal(cur_batch_x[1:], cur_batch_y[:-1])
