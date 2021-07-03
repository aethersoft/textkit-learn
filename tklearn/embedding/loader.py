from os.path import join
from typing import Text, Dict

import pandas as pd
from gensim.models import KeyedVectors

from tklearn.config import config
from tklearn.embedding import conceptnet
from tklearn.embedding.base import WordEmbedding

__all__ = [
    'load_embedding',
    'load_numberbatch',
    'load_word2vec',
]

RESOURCE_PATH = config['DEFAULT']['resource_path']


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
            join(path, filename) if path else join(RESOURCE_PATH, 'resources', filename),
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
            pd.read_hdf(join(path, filename) if path else join(RESOURCE_PATH, 'resources', filename), ),
            preprocessor=conceptnet.standardized_uri
        )
    return WordEmbedding(KeyedVectors.load_word2vec_format(
        join(path, filename) if path else join(RESOURCE_PATH, 'resources', filename),
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
