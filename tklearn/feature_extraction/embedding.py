""" Implements Embedding Transformers.
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.word_vec import WordEmbedding

__all__ = [
    'mean_embedding'
]


def mean_embedding(word_embedding: WordEmbedding) -> FunctionTransformer:
    """ Builds and returns Mean Embedding Transformer

    :param word_embedding: WordEmbedding
    :return: Mean Embedding Transformer
    """

    def _transform(X, y=None):
        lst = []
        for tokens in X:
            words = []
            for token in tokens:
                if token in word_embedding.vocab:
                    words.append(word_embedding.word_vec(token))
            if len(words) == 0:
                mean_vec = np.zeros((word_embedding.dim,))
            else:
                mean_vec = np.mean(words, axis=0)
            lst.append(mean_vec)
        return np.array(lst)

    return FunctionTransformer(_transform, validate=False)
