import json

from sklearn.preprocessing import FunctionTransformer

from tklearn.utils import resource_path
from .embedding_featurizers import *
from .lexicon_featurizers import *
from .linguistic_featurizers import *

__all__ = ['LexiconFeaturizer']


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
        extract_features = _get_featurizer(feature)
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


def _get_lexicon(name):
    resources = json.load(open(resource_path('resources.json')))
    lexicons = ['lexicons'] + resources['lexicons'][name]
    path = resource_path(*lexicons)
    return path


def _get_featurizer(name):
    """
    Gets resources from resource path.
     Resource path should contain json file indicating the resources and how to access them.
    :param name: name of the lexicon resource
    :return: path to lexicon
    """
    resources = json.load(open(resource_path('resources.json')))
    featurizer = resources['featurizers'][name]['class']
    lexicons = resources['featurizers'][name]['lexicons']
    lexicons = [_get_lexicon(l) for l in lexicons]
    if featurizer == 'PolarityCounter':
        return PolarityCounter(*lexicons)
    elif featurizer == 'SentiWordnetScorer':
        return SentiWordnetScorer(*lexicons)
    elif featurizer == 'PolarityScorer':
        return PolarityScorer(*lexicons)
    elif featurizer == 'SentimentRanking':
        fid = resources['featurizers'][name]['id']
        return SentimentRanking(*lexicons, fid)
    elif featurizer == 'LIWCExtractor':
        return LIWCExtractor(*lexicons)
    elif featurizer == 'ExtractEmbedding':
        return ExtractEmbedding(*lexicons)
    elif featurizer == 'NegationCounter':
        return NegationCounter(*lexicons)
    elif featurizer == 'EmotionLexiconScorer':
        return EmotionLexiconScorer(*lexicons)
    elif featurizer == 'SentiStrengthScorer':
        return SentiStrengthScorer(*lexicons)
    else:
        raise ModuleNotFoundError('No module named {}'.format(featurizer))
