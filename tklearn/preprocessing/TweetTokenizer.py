import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer


class TweetTokenizer(FunctionTransformer):
    _tokenizer = nltk.TweetTokenizer()

    def __init__(self):
        super(TweetTokenizer, self).__init__(self._tokenize, ' '.join, validate=False)

    @staticmethod
    def _tokenize(seq):
        return np.array([TweetTokenizer._tokenizer.tokenize(text) for text in seq])

    # @staticmethod
    # def _tokenize(seq):
    #     _tweet_nlp = TweetNLP(TWEET_NLP_PATH)
    #     tag_set = _tweet_nlp.tag(seq)
    #     tokens = [list(list(zip(*tags))[0]) for tags in tag_set]
    #     return list(zip(*tokens))[0]
