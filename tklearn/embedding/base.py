from typing import Any, Callable, NoReturn, List, Text

import numpy as np
import pandas as pd

__all__ = [
    'WordEmbedding',
]


class WordEmbedding:
    """Provides common interface for word embeddings"""

    def __init__(self, word_embedding: Any, preprocessor: Callable = None) -> NoReturn:
        """Initializer of WordEmbedding.
        Parameters
        ----------
        word_embedding : WordEmbedding
            Word Embedding (`gensim.models.KeyedVectors` or `dict`)

            preserving the tokenizing and n-grams generation steps.
        preprocessor : callable or None (default)
            Override the pre-processing (string transformation) stage while
        """
        self.preprocessor = preprocessor
        if hasattr(word_embedding, 'vocab'):
            self.vocab = set(word_embedding.vocab.keys())
        elif hasattr(word_embedding, 'index'):
            self.vocab = set(word_embedding.index.tolist())
        elif hasattr(word_embedding, 'key_to_index'):
            self.vocab = set(word_embedding.key_to_index)
        else:
            self.vocab = set(word_embedding.keys())
        self.word_embedding = word_embedding
        self.dim = 0
        for w in self.vocab:
            self.dim = len(self.word_vec(w))
            break

    def word_vec(self, word: Text) -> [List, np.array]:
        """Gets vector/embedding for the provided input word.

        Parameters
        ----------
        word :  Text
            The input word.
        Returns
        -------
            Vector representation of the input word.
        """
        if self.preprocessor is not None:
            word = self.preprocessor(word)
        if isinstance(self.word_embedding, pd.DataFrame):
            return self.word_embedding.loc[word].tolist()
        return self.word_embedding[word]

    def __getitem__(self, item: Text) -> [List, np.array]:
        """Gets the embedding of the provided word.

        Parameters
        ----------
        item: Text
            Input word.
        Returns
        -------
            Embedding of the provided word.
        """
        return self.word_vec(item)
