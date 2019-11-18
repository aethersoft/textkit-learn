""" Implements Embedding Transformers.
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.word_vec import WordEmbedding

__all__ = [
    'mean_embedding'
]


def mean_embedding(weights: WordEmbedding) -> FunctionTransformer:
    """ Builds and returns Mean Embedding Transformer

    :param weights: WordEmbedding
    :return: Mean Embedding Transformer
    """

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
                mean_vec = np.mean(words, axis=0)
            lst.append(mean_vec)
        return np.array(lst)

    return FunctionTransformer(_transform, validate=False)
