from collections import defaultdict

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.utils import get_featurizer


class LexiconFeaturizer(FunctionTransformer):
    """
    The lexicon featurizer converts a list of tokenized documents into a vector,
     in the format used for classification.
     The actual feature to be used is provided as parameter
    """

    __mem_cache = defaultdict(dict)

    def __init__(self, features=None, caching=None):
        """
        Creates a LexiconFeaturizer

        :param caching: whether to use in memory cache. useful in cross-validation tasks
        """
        super(LexiconFeaturizer, self).__init__(self._get_lexicon_features, validate=False, )
        self.caching = caching
        self.features = features

    def _get_lexicon_features(self, seq):
        """
        Transforms a sequence of token input to a vector based on lexicons.

        :param seq: Sequence of token inputs
        :return: a list of feature vectors extracted from the sequence of texts in the same order
        """
        if isinstance(self.features, str):
            return self._extract_features(seq, self.features)
        else:
            outs = [self._extract_features(seq, f) for f in self.features]
            return np.concatenate(outs, axis=1)

    def _extract_features(self, seq, feature):
        outs = []
        extract_features = get_featurizer(feature)
        for tokens in seq:
            text = ' '.join(tokens)
            if self.caching and text in LexiconFeaturizer.__mem_cache[feature]:
                temp = LexiconFeaturizer.__mem_cache[feature][text]
            else:
                temp = extract_features(tokens)
            outs.append(np.array(temp))
            if self.caching and text not in LexiconFeaturizer.__mem_cache[feature]:
                LexiconFeaturizer.__mem_cache[feature][text] = temp
        outs = np.array(outs)
        if len(outs.shape) == 1:
            outs = np.reshape(outs, (-1, 1))
        return outs
