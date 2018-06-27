from collections import Counter

import numpy as np
from keras.preprocessing import sequence
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['EmbeddingTransformer']


class RandomWordVec:
    """
    Generates random vectors for requested words and save them for future use. It is assumed that generated random
    vectors are random enough to get different vectors for each different input.
    """

    def __init__(self, dim=300):
        """
        Initializes the Random vector generator
        :param dim: dimension of generating vector
        """
        self.dim = dim
        self.word_vectors = {}

    def __getitem__(self, item):
        """
        Gets item from stored dictionary if exist else generate random vector
        :param item: word/text
        :return: generated word vector
        """
        if item in self.word_vectors.keys():
            return self.word_vectors[item]
        else:
            _rand = self.generate_rand()
            self.word_vectors[item] = _rand
            return _rand

    def generate_rand(self):
        """
        Generate and returns a  random value
        :return: a random valued vector of size self.dim
        """
        return np.random.rand(self.dim)


class ZeroWordVec:
    """
    Return Zero vectors of size dim as provided in const. Implemented for consistency reasons.
    """

    def __init__(self, dim=300):
        """
        Initializes the Random vector generator
        :param dim: dimension of generating vector
        """
        self.dim = dim

    def __getitem__(self, item):
        """
        Gets item from stored dictionary if exist else generate random vector
        :param item: word/text
        :return: generated word vector
        """
        return [0 for _ in range(self.dim)]


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_vectors, vocab=None, word_features=None, vocab_size=None, pad_sequences=False,
                 default=None, output='matrix'):
        """
        Scikit-learn transformer like interface for tweet tokenizing

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
        self.output = output

    def fit(self, X, *_):
        """
        Fits on the training set - extracts words and word frequencies

        :param X: Training sentence vector
        :return: self
        """
        # build word frequency
        assert self.word_vectors is not None, 'Failed to extract word embeddings without proper word vector.'
        word_freq = Counter()
        if self.vocab is None:
            for tokens in X:
                for token in tokens:
                    word_freq[token] += 1
        else:
            for tokens in self.vocab:
                if isinstance(tokens, list):
                    for token in tokens:
                        word_freq[token] += 1
                else:
                    word_freq[tokens] += 1
        # Build word index
        self.word_index_ = dict()
        if self.vocab_size is None:
            frq_words = word_freq.most_common()
        else:
            frq_words = word_freq.most_common(self.vocab_size - 1)
        idx_ = 1
        for token, _ in frq_words:
            self.word_index_[token] = idx_
            idx_ += 1
        self._build_embedding_matrix(X)
        self.word_vectors = None
        self.vocab = None
        return self

    def transform(self, X, *_):
        """
        Transforms the provided data-set and outputs Sequence of word indexes observed when training.
        All the unobserved words (words not in vocabulary) are ignored.

        :param X: Input sentence vector.
        :return: Sequence of word indexes of each sentence as am array
        """
        sequences = []
        for tokens in X:
            token_indexes = []
            for token in tokens:
                try:
                    token_indexes += [self.word_index_[token]]
                except KeyError:
                    #  Removes word indexes not identified in fitting phrase
                    #  "Ignoring the words not found"
                    #  Use {{self.default}} to map words not in embeddings to word vector
                    pass
            sequences += [token_indexes]
        if self.pad_sequences:
            assert isinstance(self.pad_sequences, int), \
                'Invalid input for pad_sequence. Please provide an integer value for `pad_sequence` parameter.'
            sequences = sequence.pad_sequences(sequences, maxlen=self.pad_sequences)
        if self.output == 'matrix':
            return {
                'tokens': np.array(sequences),
                'embedding_matrix': self.embedding_matrix_,
                'word_index': self.word_index_,
            }
        elif self.output == 'average':
            dim = self.embedding_matrix_.shape[1]
            result = []
            for x in X:
                avg = np.average([self.embedding_matrix_[e] for e in x], axis=0) if len(x) > 0 else np.zeros(dim)
                result.append(avg)
            return np.array(result)
        else:
            dim = self.embedding_matrix_.shape[1]
            result = []
            for x in X:
                avg = np.sum([self.embedding_matrix_[e] for e in x], axis=0) if len(x) > 0 else np.zeros(dim)
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

    def _build_embedding_matrix(self, X):
        """
        Builds embedding matrix

        :param X: input data
        :return: embedding matrix
        """
        self.embedding_matrix_ = None
        _word_index = {}
        for word, idx in self.word_index_.items():
            embedding_vector = self._generate_word_vector(word)
            if embedding_vector is not None:
                if self.embedding_matrix_ is None:
                    self.embedding_matrix_ = np.zeros((len(self.word_index_) + 1, len(embedding_vector)))
                try:
                    self.embedding_matrix_[idx] = embedding_vector
                except ValueError:
                    raise ValueError(
                        'Error setting embedding for word \'{}\'. The expected size is {} and output size is {}'
                            .format(word, len(self.embedding_matrix_[idx]), len(embedding_vector)))
                _word_index[word] = idx
            else:
                #  ignore words not in embeddings
                pass
        self.word_index_ = _word_index

    def _generate_word_vector(self, word):
        """
        Generates feature vector
        :param word: a word to generate features for
        :return: word vector including extra features
        """
        out = []
        if self.default != 'skip':
            try:
                out = self.word_vectors[word]
            except KeyError:
                out = None
            if out is None:
                if self.default == 'random':
                    self.default = RandomWordVec(self.word_vectors.vector_size)
                elif self.default == 'zero':
                    self.default = ZeroWordVec(self.word_vectors.vector_size)
                elif self.default == 'ignore' or self.default is None:
                    return None
                out = self.default[word]
        if self.word_features is not None:
            extra_features = self.word_features(word)
            extra = np.array(extra_features).astype(float)
            out = np.append(out, extra)
        return out
