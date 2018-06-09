import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.twitter import TweetNLP


class TweetTokenizer(FunctionTransformer):
    _tokenizer = nltk.TweetTokenizer()

    def __init__(self, preprocessor=None, tokenizer=None, stopwords=None, vocabulary=None):
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        super(TweetTokenizer, self).__init__(self.tokenize, ' '.join, validate=False)

    def tokenize(self, seq):
        if self.preprocessor is not None:
            seq = [self.preprocessor(s) for s in seq]
        if self.tokenizer is None:
            output = [TweetTokenizer._tokenizer.tokenize(text) for text in seq]
        elif isinstance(self.tokenizer, TweetNLP):
            tag_set = self.tokenizer.tag(seq)
            output = [list(list(zip(*tags))[0]) for tags in tag_set]
        else:
            output = [self.tokenizer.tokenize(text) for text in seq]
        if self.stopwords is not None:
            output = [[token for token in tweet if token not in self.stopwords] for tweet in output]
        if self.vocabulary is not None:
            output = [[token for token in tweet if token in self.vocabulary] for tweet in output]
        return np.array(output)
