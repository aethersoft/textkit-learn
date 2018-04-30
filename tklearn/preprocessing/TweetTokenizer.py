import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.twitter.TwitterNLP import TweetNLP


class TweetTokenizer(FunctionTransformer):
    _tokenizer = nltk.TweetTokenizer()

    def __init__(self, tweet_nlp_path=None):
        self.tweet_nlp_path = tweet_nlp_path
        super(TweetTokenizer, self).__init__(self._tokenize, ' '.join, validate=False)

    def _tokenize(self, seq):
        if self.tweet_nlp_path is None:
            return np.array([TweetTokenizer._tokenizer.tokenize(text) for text in seq])
        else:
            _tweet_nlp = TweetNLP(self.tweet_nlp_path)
            tag_set = _tweet_nlp.tag(seq)
            tokens = [list(list(zip(*tags))[0]) for tags in tag_set]
            return tokens
