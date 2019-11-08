from os.path import join
from typing import Callable, NoReturn, Any, List, Text, Dict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from tklearn.configs import configs
from tklearn.text import _numberbatch


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


def load_word2vec(filename: Text = 'GoogleNews-vectors-negative300.bin.gz', path: Text = None) -> WordEmbedding:
    """Loads binary word embedding stored at provided location.

    By default this will try to load `GoogleNews-vectors-negative300.bin.gz` from project resource folder.

    :param filename: Text
        Name of word embedding file.
    :param path: Text
        Path to word embedding file.
    :return: The GoogleNews-vectors-negative300 WordEmbedding.
    """
    return WordEmbedding(
        KeyedVectors.load_word2vec_format(
            join(path, filename) if path else join(configs['OLANG_PATH'], 'resources', filename),
            binary=True
        )
    )


def load_numberbatch(filename: Text = 'numberbatch-17.06-mini.h5', path: Text = None) -> WordEmbedding:
    """Loads numberbatch embedding stored at provided location.

    :param filename: Text
        Name of word embedding file.
    :param path: Text
        Path to numberbatch embedding file.
    :return: The Numberbatch WordEmbedding.
    """
    if filename.endswith('.h5'):
        return WordEmbedding(
            pd.read_hdf(join(path, filename) if path else join(configs['OLANG_PATH'], 'resources', filename), ),
            preprocessor=_numberbatch.standardized_uri
        )
    return WordEmbedding(KeyedVectors.load_word2vec_format(
        join(path, filename) if path else join(configs['OLANG_PATH'], 'resources', filename),
        binary=False
    ))


def load_embedding(d: Dict) -> WordEmbedding:
    """Loads word embedding from a dict.

    :type d: Dict
        A dictionary of words mapping to word vectors.
    :return: WordEmbedding.
    """
    return WordEmbedding(d)
