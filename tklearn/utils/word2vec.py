from gensim.models import KeyedVectors

__all__ = [
    'load_word2vec'
]


class Word2Vec:
    def __init__(self, w2v, binary=True, unicode_errors='ignore'):
        self.w2v = w2v

    @property
    def vector_size(self):
        return self.w2v.vector_size

    def __getitem__(self, item):
        return self.w2v[item]

    def __deepcopy__(self, memodict={}):
        obj = Word2Vec(self.w2v)
        return obj

    def __copy__(self):
        obj = Word2Vec(self.w2v)
        return obj


def load_word2vec(path, binary=True, unicode_errors='ignore'):
    """
    Loads and returns word2vec
    :param binary: a boolean indicating whether the data is in binary word2vec format.
    :param unicode_errors: default 'ignore', a string suitable to be passed as the errors argument to the unicode()
            (Python 2.x) or str() (Python 3.x) function.
    :param path: path to word2vec file
    :return: Binary Word2Vec
    """
    w2v = KeyedVectors.load_word2vec_format(path, binary=binary, unicode_errors=unicode_errors)
    return Word2Vec(w2v, binary=binary, unicode_errors=unicode_errors)
