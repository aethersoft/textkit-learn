from gensim.models import KeyedVectors

__all__ = [
    'load_word2vec'
]


def load_word2vec(path, binary=True, unicode_errors='ignore'):
    """
    Loads and returns word2vec
    :param binary: a boolean indicating whether the data is in binary word2vec format.
    :param unicode_errors: default 'ignore', a string suitable to be passed as the errors argument to the unicode()
            (Python 2.x) or str() (Python 3.x) function.
    :param path: path to word2vec file
    :return: Binary Word2Vec
    """
    return KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
