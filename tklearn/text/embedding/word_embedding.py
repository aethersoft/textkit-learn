import csv

import gensim
import numpy as np
import pandas as pd

__all__ = [
    'load_word2vec',
    'load_glove'
]


class WordEmbedding:
    def __init__(self, word_embedding, vector_size):
        self.word_embedding = word_embedding
        self._vector_size = vector_size

    @property
    def vector_size(self):
        return self._vector_size

    def __getitem__(self, item):
        try:
            if isinstance(self.word_embedding, pd.DataFrame):
                return np.array(self.word_embedding.loc[item].as_matrix())
            return self.word_embedding[item]

        except KeyError:
            return None

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
        model = gensim.models.Word2Vec.load('300features_40minwords_10context.txt')
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
