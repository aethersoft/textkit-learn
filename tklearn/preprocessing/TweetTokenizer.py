import nltk
import numpy as np
from gensim.utils import simple_preprocess
from sklearn.preprocessing import FunctionTransformer


class TweetTokenizer(FunctionTransformer):
    _tokenizer = nltk.TweetTokenizer()

    def __init__(self, tweet_nlp=None):
        self.tweet_nlp = tweet_nlp
        super(TweetTokenizer, self).__init__(self._tokenize, ' '.join, validate=False)

    def _tokenize(self, seq):
        if self.tweet_nlp is None:
            return np.array([TweetTokenizer._tokenizer.tokenize(text) for text in seq])
        elif self.tweet_nlp == 'simple_preprocess':
            return np.array([simple_preprocess(text) for text in seq])
        else:
            tag_set = self.tweet_nlp.tag(seq)
            tokens = [list(list(zip(*tags))[0]) for tags in tag_set]
            return tokens
