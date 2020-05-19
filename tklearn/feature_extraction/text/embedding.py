""" Implements Embedding Transformers.
"""
from typing import Text

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.embedding.loader import WordEmbedding

__all__ = [
    'make_embedding_transformer'
]


def make_embedding_transformer(weights: WordEmbedding, method: Text = 'average') -> FunctionTransformer:
    """ Builds and returns Mean Embedding Transformer

    Parameters
    ----------
    weights
        WordEmbedding

    method
        Strategy to use to get embedding for sentences. Should be one of ['sum', 'average']

    Returns
    -------
        Mean Embedding Transformer
    """

    # noinspection PyPep8Naming
    def _transform(X, y=None):
        lst = []
        for tokens in X:
            words = []
            for token in tokens:
                try:
                    words.append(weights.word_vec(token))
                except KeyError as _:
                    pass
            if len(words) == 0:
                mean_vec = np.zeros((weights.dim,))
            else:
                if method == 'sum':
                    mean_vec = np.sum(words, axis=0)
                else:
                    mean_vec = np.mean(words, axis=0)
            lst.append(mean_vec)
        return np.array(lst)

    return FunctionTransformer(_transform, validate=False)
