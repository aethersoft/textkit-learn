from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EmbeddingAverager(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Averages embedding over a sequence of words identified by index. Should come after {{EmbeddingExtractor}}.
        """
        pass

    def fit(self, X, *_):
        """
        No fitting required.

        :param X: Training sentence vector
        :return: self
        """
        return self

    def transform(self, X, *_):
        """
        Computes the average of word embeddings as identified by the sequence of encoded indexes.

        :param X: Input sentence vector.
        :return: Input sequence
        """
        embedding_matrix = X['embedding_matrix']
        X = X['tokens']
        result = []
        for x in X:
            result += [np.average([embedding_matrix[e] for e in x], axis=0)]
        return np.array(result)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Performs Fitting and Transformation sequentially

        :param X: Input sentence vector.
        :param y: [ignored - added for consistency]
        :param fit_params: [ignored - added for consistency]
        :return: Sequence of word indexes of each sentence as am array
        """
        return self.fit(X).transform(X, y)
