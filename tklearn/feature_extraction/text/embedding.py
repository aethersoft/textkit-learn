""" Implements Embedding Transformers.
"""
from typing import Text

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tklearn.embedding.base import WordEmbedding

__all__ = [
    'make_embedding_vectorizer'
]


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """EmbeddingVectorizer"""

    def __init__(self, weights, method):
        self.weights = weights
        self.method = method

    def fit(self, text: list, y=None) -> 'EmbeddingVectorizer':
        return self

    def transform(self, raw_documents, y=None):
        """Transform input documents.
        Parameters
        ----------
        raw_documents : array-like, shape (n_samples,)
            Input text.
        y: None
            Value is ignored.
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        lst = []
        for tokens in raw_documents:
            words = []
            for token in tokens:
                try:
                    words.append(self.weights.word_vec(token))
                except KeyError as _:
                    pass
            if len(words) == 0:
                mean_vec = np.zeros((self.weights.dim,))
            else:
                if hasattr(self.method, '__call__'):
                    mean_vec = [self.method() for x in words]
                elif self.method == 'sum':
                    mean_vec = np.sum(words, axis=0)
                else:
                    mean_vec = np.mean(words, axis=0)
            lst.append(mean_vec)
        return np.array(lst)


def make_embedding_vectorizer(weights: WordEmbedding, method: Text = 'average') -> EmbeddingVectorizer:
    """Builds and returns Embedding Vectorizer
    Parameters
    ----------
    weights
        WordEmbedding
    method
        Strategy to use to get embedding for sentences. Should be one of ['sum', 'average']
    Returns
    -------
        EmbeddingVectorizer
    """
    return EmbeddingVectorizer(weights, method)
