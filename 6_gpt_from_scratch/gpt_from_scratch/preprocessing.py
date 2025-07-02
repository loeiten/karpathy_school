"""Module for loading and preprocessing data."""

from typing import List, Tuple
from enum import Enum

import torch
from gpt_from_scratch.utils.paths import get_data_path

from gpt_from_scratch.utils.logger import Logger
from gpt_from_scratch import DEVICE


class DataSet(Enum):
    """Class containing enumeration of the datasets."""

    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"


class DataPreprocessor:
    """Class for preprocessing the data."""

    def __init__(self, context_length: int) -> None:
        """
        Create the raw data.

        Args:
            context_length (int): The context length, a.k.a the block size
        """
        self.logger = Logger.get_logger()
        data_path = get_data_path()
        raw_text = data_path.open("r").read()
        tokenizer = CharTokenizer(corpus=raw_text)
        raw_data = torch.tensor(tokenizer.encode(raw_text), device=DEVICE)
        self.logger.debug(f"Full data shape: {raw_data.shape}")
        # Split 90 % train, 10 % validation
        n = int(0.9 * len(raw_data))
        self.train_split = raw_data[:n]
        self.validation_split = raw_data[n:]
        self.logger.debug(f"n tokens train_split: {len(self.train_split)}")
        self.logger.debug(f"n tokens validation_split: {len(self.validation_split)})")
        self.context_length = context_length
        self.logger.debug(f"Setting context length to: {self.context_length})")

    def get_batch(
        self, sample_from: DataSet, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch randomly sampled from the raw data.

        A batch contains several examples (rows).
        Assuming that we have the example [24, 43, 58,  5, 57,  1, 46, 43]
        And the corresponding targets     [43, 58,  5, 57,  1, 46, 43, 39]
        Notice how the targets are shifted 1 from the input example

        Using this example, due to how the attention mechanism masks out tokens
        we will have:

        When the prompt is tensor([24]), predict 43
        When the prompt is tensor([24, 43]), predict 58
        When the prompt is tensor([24, 43, 58]), predict 5
        When the prompt is tensor([24, 43, 58,  5]), predict 57
        When the prompt is tensor([24, 43, 58,  5, 57]), predict 1
        When the prompt is tensor([24, 43, 58,  5, 57,  1]), predict 46
        When the prompt is tensor([24, 43, 58,  5, 57,  1, 46]), predict 43
        When the prompt is tensor([24, 43, 58,  5, 57,  1, 46, 43]), predict 39

        However, this is done in parallel for the whole example.
        I.e. we don't have to feed in first [24], then [24, 43] and so on during
        training.
        However, this is what we do during inference.
        The first token will predict the second token.
        The first token + the second token will predict the third token

        Note that the forward pass in our model will work on even one token.
        There are two reasons for this:
        1. All tokens are embedded
        2. We are doing batched matrix multiplication

        1. Embeddings
        =============

        Imagine our embedding is

        c =
        [[a, b, c],
         [d, e, f]]

        Imagine further that our indices matrix are

        ind =
        [[1, 0, 1, 1],
         [0, 0, 0, 0],
         ...
        ]

        We get:

        ind[0,0] = 1 => [d, e, f]
        ind[0,1] = 0 => [a, b, c]
        ind[0,2] = 1 => [d, e, f]
        ind[0,3] = 1 => [d, e, f]

        ind[1,0] = 0 => [a, b, c]
        ind[1,1] = 0 => [a, b, c]
        ind[1,2] = 0 => [a, b, c]
        ind[1,3] = 0 => [a, b, c]

        ...

        Hence

        c[ind] =
        [[[d, e, f],
          [a, b, c],
          [d, e, f],
          [d, e, f]],
         [[a, b, c],
          [a, b, c],
          [a, b, c],
          [a, b, c]],
         ...
        ]

        The following example from advanced indexing of numpy is instructive:
        >>> x = np.array([[ 0,  1,  2],
        ...       [ 3,  4,  5],
        ...       [ 6,  7,  8],
        ...       [ 9, 10, 11]])
        >>> rows = np.array([[0, 0],
        ...                  [3, 3]])
        >>> columns = np.array([[0, 2],
        ...                     [0, 2]])
        >>> x[rows, columns]
            array([[ 0,  2],
                   [ 9, 11]])

        I.e. a specific ind element will pick out one row of c.
        This row will have the size of the embedding.
        When moving to the next element (in the same row) of ind,
        the row number will be picked out and stacked underneath the row picked
        out by the previous element.
        When moving to a new row in ind, we will start on a new sub-matrix.
        No matter the shape of ind, c[ind] will have the embedding size as the
        last dimension.
        We can always add more "fake dimensions" to ind to make it have the same
        dimensionality as when we trained.
        E.g. [[[1]]] instead of [1]

        2. Batched matrix multiplication
        ================================
        When we are multiplying together multidimensional matrices, we are
        only multiplying together the last dimensions.

        I.e. if we have c=a@b
        where a have dimensionality     [d0, d1, d2, ..., e, f]
        and b have dimensionality       [d0, d1, d2, ..., f, g]
        then c will have dimensionality [d0, d1, d2, ..., e, g]

        I.e. we are multiplying the innermost matrix for all the elements in the
        "super" matrix

        NOTE: Since we have no start and end tokens, we do not need any padding.
              When we draw a batch we will draw random context lengths
              characters from the text.
              During inference, we will stop after max_new_tokens
              Due to this random draw, we also don't shuffle the dataset.

        Args:
            sample_from (DataSet): Data set to sample from
            batch_size (int): The batch size

        Returns:
            torch.Tensor: The input_data of shape (batch_size, context_length)
            torch.Tensor: The targets of shape (batch_size, context_length)

        Example:
            >>> dp = DataPreprocessor(block_size)
            >>> xb, yb = dp.get_batch(sample_from=DataSet.TRAIN, batch_size=4)
            >>> xb
                tensor([[24, 43, 58,  5, 57,  1, 46, 43],
                        [44, 53, 56,  1, 58, 46, 39, 58],
                        [52, 58,  1, 58, 46, 39, 58,  1],
                        [25, 17, 27, 10,  0, 21,  1, 54]])
            >>> yb
                tensor([[43, 58,  5, 57,  1, 46, 43, 39],
                        [53, 56,  1, 58, 46, 39, 58,  1],
                        [58,  1, 58, 46, 39, 58,  1, 46],
                        [17, 27, 10,  0, 21,  1, 54, 39]])
        """
        data = (
            self.train_split if sample_from == DataSet.TRAIN else self.validation_split
        )
        # We need to subtract the context length as we will stack examples with
        # context length to the batch
        start_idxs = torch.randint(
            low=0, high=(len(data) - self.context_length), size=(batch_size,)
        )
        input_data = torch.stack(
            [
                data[start_idx : start_idx + self.context_length]
                for start_idx in start_idxs
            ]
        )
        targets = torch.stack(
            [
                data[start_idx + 1 : start_idx + self.context_length + 1]
                for start_idx in start_idxs
            ]
        )
        return input_data, targets


class CharTokenizer:
    """Class for character tokenize the corpus"""

    def __init__(self, corpus: str) -> None:
        """
        Create the tokenizer.

        NOTE: We have no special start or end tokens

        Args:
            corpus (str): The corpus to tokenize from
        """
        self.logger = Logger.get_logger()
        self.tokens = sorted(list(set(corpus)))
        self.vocab_size = len(self.tokens)
        self.logger.debug(f"Vocab size {self.vocab_size}")
        self.logger.debug(f"Tokens {self.tokens}")
        self.token_to_index = {char: index for index, char in enumerate(self.tokens)}
        self.index_to_token = {index: char for index, char in enumerate(self.tokens)}

    def encode(self, str_message: str) -> List[int]:
        """Encode a message.

        Args:
            str_message (str): The message to encode

        Returns:
            List[int]: The encoded message
        """
        return [self.token_to_index[char] for char in str_message]

    def decode(self, token_message: List[int]) -> str:
        """Decode a message.

        Args:
            message (str): The encoded message

        Returns:
            List[int]: The decoded message
        """
        return "".join([self.index_to_token[token] for token in token_message])
