import numpy as np
import six
from keras.preprocessing import sequence
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_index=None, tokenizer=None, preprocess=None, padding=0, output=None,
                 weight_mat=None, *args, **kwargs):
        self.word_index = word_index
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.output = output
        self.weight_mat = weight_mat
        self.padding = padding
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
        tokenize = self.tokenizer.tokenize if hasattr(self.tokenizer, 'tokenize') else self.tokenizer
        for tokens in tokenize(pt):
            seq.append([self.word_index[w] for w in tokens if w in self.word_index])
        if self.output is None or self.output.startswith('seq'):
            if self.padding:
                assert isinstance(self.padding, int), \
                    'Invalid input for `padding`. Please provide an integer value for `padding` parameter.'
                seq = sequence.pad_sequences(seq, maxlen=self.padding, truncating='post')
            return seq
        if self.output.startswith('avg'):
            dim = len(self.weight_mat[0])
            result = []
            for x in seq:
                avg = np.average([self.weight_mat[e] for e in x], axis=0) if len(x) > 0 else np.zeros(dim)
                result.append(avg)
            return np.array(result)
        if self.output.startswith('sum'):
            dim = len(self.weight_mat[0])
            result = []
            for x in seq:
                avg = np.sum([self.weight_mat[e] for e in x], axis=0) if len(x) > 0 else np.zeros(dim)
                result.append(avg)
            return np.array(result)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
