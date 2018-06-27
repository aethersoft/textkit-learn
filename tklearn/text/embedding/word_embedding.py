import csv
import gzip
import re

import gensim
import numpy as np
import pandas as pd
import six

__all__ = [
    'load_word2vec',
    'load_glove'
]


class WordEmbedding:
    def __init__(self, word_embedding, vector_size):
        self.word_embedding = word_embedding
        self._vector_size = vector_size
        self.secondary_word_embeddings = []

    @property
    def vector_size(self):
        return self._vector_size

    @property
    def vocabulary(self):
        if isinstance(self.word_embedding, pd.DataFrame):
            raise NotImplementedError
        if isinstance(self.word_embedding, dict):
            return set(self.word_embedding.keys())
        vocab = set(self.word_embedding.vocab.keys())
        for e in self.secondary_word_embeddings:
            vocab.update(e.vocabulary)
        return vocab

    def append(self, e):
        """
        Appends another word embedding of same size (required if there is an extension to another word embedding)
        :param e:
        :return:
        """
        if self.vector_size != e.vector_size:
            raise ValueError(
                'Invalid embedding dim, expected {} found {}.'.format(self.vector_size, e.vector_size))
        self.secondary_word_embeddings.append(e)

    def __getitem__(self, item):
        try:
            if isinstance(self.word_embedding, pd.DataFrame):
                return np.array(self.word_embedding.loc[item].as_matrix())
            return self.word_embedding[item]
        except KeyError:
            output = None
            for e in self.secondary_word_embeddings:
                output = e[item]
                if output is not None:
                    break
            return output

    def __deepcopy__(self, memodict={}):
        obj = WordEmbedding(self.word_embedding, self.vector_size)
        return obj

    def __copy__(self):
        obj = WordEmbedding(self.word_embedding, self.vector_size)
        return obj


def load_word2vec(data_file, binary=True, unicode_errors='ignore', verbose=False):
    """
    Loads and returns word2vec
    :param binary: a boolean indicating whether the data is in binary word2vec format.
    :param unicode_errors: default 'ignore', a string suitable to be passed as the errors argument to the unicode()
            (Python 2.x) or str() (Python 3.x) function.
    :param data_file: path to word2vec file
    :return: Binary Word2Vec
    """
    if not binary:
        model = gensim.models.Word2Vec.load(data_file)
        w2v = model.wv
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(data_file, binary=binary, unicode_errors=unicode_errors)
    return WordEmbedding(w2v, w2v.vector_size)


def load_glove(data_file):
    """
    Loads Glove vectors to {{ WordEmbedding }} structure
    :param data_file:
    :return:
    """
    temp = pd.read_table(data_file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
    return WordEmbedding(temp, temp.shape[1])


def load_embedding(data_file, word_first=False, leave_head=False):
    """
    Creates a map from words to word embeddings (in text format)
    :return: embedding map
    """
    with gzip.open(data_file, 'rb') as f:
        lines = f.read().decode('utf8').splitlines()
    if leave_head:
        lines = lines[1:]
    lexicon_map = {}
    for l in lines:
        splits = re.split('[\t ]', l)
        if word_first:
            lexicon_map[splits[0]] = np.asarray(splits[1:], dtype='float32')
        else:
            lexicon_map[splits[-1]] = np.asarray(splits[:-1], dtype='float32')
    dim = len(six.next(six.itervalues(lexicon_map)))
    return WordEmbedding(lexicon_map, dim)
