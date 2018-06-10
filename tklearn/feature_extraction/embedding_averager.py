import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .embedding_extractor import EmbeddingExtractor

__all__ = ['EmbeddingAverager']


class EmbeddingAverager(BaseEstimator, TransformerMixin):
    def __init__(self, word_vectors, vocab=None, word_features=None, vocab_size=None, pad_sequences=False,
                 default=None):
        """
        Computes embedding average provided the word vectors

        :param word_vectors: a word to embedding mapper
        :param vocab: a list of sentences to extract the vocabulary (by tokenizing) or list of words.
                      If this is None, defaults to extracting vocabulary using dataset provided when fitting.
        :param word_features: a callable with one string parameter to generate features based on input
        :param vocab_size: maximum number of words in vocabulary
        :param pad_sequences: integer indicating the size of the sequence vector
        :param default: default word_vector / generator for missing word values
        """
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.pad_sequences = pad_sequences
        self.word_vectors = word_vectors
        self.default = default
        self.word_features = word_features

    def fit(self, X, *_):
        """
        No fitting required.

        :param X: Training sentence vector
        :return: self
        """
        self.embeddings_ = EmbeddingExtractor(self.word_vectors, self.vocab, self.word_features, self.vocab_size,
                                              self.pad_sequences, self.default)
        self.embeddings_.fit(X)
        return self

    def transform(self, X, *_):
        """
        Computes the average of word embeddings as identified by the sequence of encoded indexes.

        :param X: Input sentence vector.
        :return: Input sequence
        """
        X = self.embeddings_.transform(X)
        embedding_matrix = X['embedding_matrix']
        X = X['tokens']
        dim = embedding_matrix.shape[1]
        result = []
        for x in X:
            avg = np.average([embedding_matrix[e] for e in x], axis=0) if len(x) > 0 else np.zeros(dim)
            result.append(avg)
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
