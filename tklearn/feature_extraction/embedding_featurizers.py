import gzip
import re

import numpy as np


class ExtractEmbedding:
    def __init__(self, lexicon_path, word_first=False, leave_head=False):
        self.lexicon_path = lexicon_path
        self.word_first = word_first
        self.leave_head = leave_head
        self.embedding_map = self.load_embeddings()

    def __call__(self, tokens):
        """
        Averaging word embeddings
        """
        if len(self.embedding_map.keys()) > 0:
            dim = len(list(self.embedding_map.values())[0])
        else:
            dim = 0
        sum_vec = np.zeros(shape=(dim,))
        for token in tokens:
            if token in self.embedding_map:
                vec = self.embedding_map[token]
                assert len(vec) == dim, 'Invalid length for embedding provided. Observed {} expected {}' \
                    .format(len(vec), dim)
                sum_vec = sum_vec + vec
        denom = len(tokens)
        sum_vec = sum_vec / denom
        return sum_vec

    def load_embeddings(self):
        """
        Creates a map from words to word embeddings
        :return: embedding map
        """
        with gzip.open(self.lexicon_path, 'rb') as f:
            lines = f.read().decode('utf8').splitlines()
        if self.leave_head:
            lines = lines[1:]
        lexicon_map = {}
        for l in lines:
            splits = re.split('[\t ]', l)
            if self.word_first:
                lexicon_map[splits[0]] = np.asarray(splits[1:], dtype='float32')
            else:
                lexicon_map[splits[-1]] = np.asarray(splits[:-1], dtype='float32')
        return lexicon_map
