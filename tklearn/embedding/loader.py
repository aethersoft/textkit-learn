from os.path import join
from typing import Any, Callable, NoReturn, List, Text, Dict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from tklearn.config import config
from tklearn.embedding import conceptnet

# noinspection SpellCheckingInspection
__all__ = [
    'WordEmbedding',
    'load_embedding',
    'load_numberbatch',
    'load_word2vec',
]


class WordEmbedding:
    """Provides common interface for word embeddings"""

    def __init__(self, word_embedding: Any, preprocessor: Callable = None) -> NoReturn:
        """ Initializer of WordEmbedding.

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
        """ Gets vector/embedding for the provided input word.

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
        return self.word_vec(item)


def load_word2vec(filename: Text = 'GoogleNews-vectors-negative300.bin.gz', path: Text = None) -> WordEmbedding:
    """ Loads binary word embedding stored at provided location.

        By default this will try to load `GoogleNews-vectors-negative300.bin.gz` from project resource folder.

    Parameters
    ----------
    filename : Text
        Name of word embedding file.

    path : Text
        Path to word embedding file.

    Returns
    -------
        The GoogleNews-vectors-negative300 WordEmbedding.
    """
    return WordEmbedding(
        KeyedVectors.load_word2vec_format(
            join(path, filename) if path else join(config['RESOURCE_PATH'], 'resources', filename),
            binary=True
        )
    )


# noinspection SpellCheckingInspection
def load_numberbatch(filename: Text = 'numberbatch-17.06-mini.h5', path: Text = None) -> WordEmbedding:
    """ Loads numberbatch embedding stored at provided location.

    Parameters
    ----------
    filename : Text
        Name of word embedding file.

    path : Text
        Path to numberbatch embedding file.

    Returns
    -------
        The Numberbatch WordEmbedding.
    """
    if filename.endswith('.h5'):
        return WordEmbedding(
            pd.read_hdf(join(path, filename) if path else join(config['RESOURCE_PATH'], 'resources', filename), ),
            preprocessor=conceptnet.standardized_uri
        )
    return WordEmbedding(KeyedVectors.load_word2vec_format(
        join(path, filename) if path else join(config['RESOURCE_PATH'], 'resources', filename),
        binary=False
    ))


def load_embedding(d: Dict) -> WordEmbedding:
    """ Loads word embedding from a dict.

    Parameters
    ----------
    d : Dict
        A dictionary of words mapping to word vectors.

    Returns
    -------
    word_embedding
        WordEmbedding.
    """
    return WordEmbedding(d)
