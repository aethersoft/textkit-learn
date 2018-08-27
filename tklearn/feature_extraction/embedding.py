import numpy as np
import six
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_index=None, tokenizer=None, preprocess=None, output=None, weight_mat=None, *args, **kwargs):
        self.word_index = word_index
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.output = output
        self.weight_mat = weight_mat
        self.verify()

    def verify(self):
        if self.output == 'average':
            if self.weight_mat is None:
                raise TypeError('Weight Matrix can\'t be null when average embedding is required.')
            if len(self.weight_mat) < 1:
                raise ValueError('The weight matrix should contain at least one row.')

    def fit(self, X, y=None, *args, **kwargs):
        # triggers a parameter validation
        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
        seq = []
        pt = self.preprocess(X) if self.preprocess is not None else X
        for tokens in self.tokenizer(pt):
            seq.append([self.word_index[w] for w in tokens if w in self.word_index])
        if self.output is None or self.output.startswith('seq'):
            return seq
        if self.output.startswith('avg'):
            dim = len(self.weight_mat[0])
            avg_emb = np.average([self.weight_mat[e] for e in seq], axis=0) if len(seq) > 0 else np.zeros(dim)
            return avg_emb
        if self.output.startswith('sum'):
            dim = len(self.weight_mat[0])
            sum_emb = np.sum([self.weight_mat[e] for e in seq], axis=0) if len(seq) > 0 else np.zeros(dim)
            return sum_emb

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
