from typing import Any, Callable, NoReturn, Text, List
import numpy as np
import pandas as pd


class WordEmbedding:
    """Provides common interface for word embeddings"""

    def __init__(self, word_embedding: Any, preprocessor: Callable = None) -> NoReturn:
        """Initializer of WordEmbedding.

        :type preprocessor: callable or None (default)
            Override the preprocessing (string transformation) stage while
            preserving the tokenizing and n-grams generation steps.
        :param word_embedding: Word Embedding (`gensim.models.KeyedVectors` or `dict`)
        """
        self.preprocessor = preprocessor
        if hasattr(word_embedding, 'vocab'):
            self.vocab = set(word_embedding.vocab.keys())
        elif hasattr(word_embedding, 'index'):
            self.vocab = set(word_embedding.index.tolist())
        else:
            self.vocab = set(word_embedding.keys())
        self.word_embedding = word_embedding
        self.dim = 0
        for w in self.vocab:
            self.dim = len(self.word_vec(w))
            break

    def word_vec(self, word: Text) -> [List, np.array]:
        """Gets vector/embedding for the provided input word.

        :param word: Text
            The input word.
        :return: Vector representation of the input word.
        """
        if self.preprocessor is not None:
            word = self.preprocessor(word)
        if isinstance(self.word_embedding, pd.DataFrame):
            return self.word_embedding.loc[word].tolist()
        return self.word_embedding[word]

    def __getitem__(self, item: Text) -> [List, np.array]:
        return self.word_vec(item)
