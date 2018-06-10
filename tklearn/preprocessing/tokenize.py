import re

import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.twitter import TweetNLP

__all__ = [
    'Tokenizer',
    'TweetTokenizer',
    'DictionaryTokenizer',
]


class Tokenizer(FunctionTransformer):
    """
    Scikit-learn compatible tokenizer interface.
    """

    def __init__(self, tokenizer, preprocessor=None, stopwords=None, vocabulary=None):
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        super(Tokenizer, self).__init__(self._tokenize, ' '.join, validate=False)

    def _tokenize(self, itr):
        if self.preprocessor is not None:
            itr = [self.preprocessor(s) for s in itr]
        output = self.tokenize(itr)
        if self.stopwords is not None:
            output = [[token for token in tweet if token not in self.stopwords] for tweet in output]
        if self.vocabulary is not None:
            output = [[token for token in tweet if token in self.vocabulary] for tweet in output]
        return np.array(output)

    def tokenize(self, itr):
        if not hasattr(self.tokenizer, '__call__'):
            raise TypeError('\'{}\' object is not callable.'.format(self.tokenizer.__class__.__name__))
        return [self.tokenizer(text) for text in itr]


class TweetTokenizer(Tokenizer):
    """
    Tweet tokenizer implementation.
    """

    def __init__(self, tokenizer=None, preprocessor=None, stopwords=None, vocabulary=None):
        super(TweetTokenizer, self).__init__(tokenizer, preprocessor, stopwords, vocabulary)
        self.initialize()

    def initialize(self):
        if self.tokenizer is None:
            self.tokenizer = nltk.TweetTokenizer()

    def tokenize(self, itr):
        if isinstance(self.tokenizer, TweetNLP):
            tag_set = self.tokenizer.tag(itr)
            return [list(list(zip(*tags))[0]) for tags in tag_set]
        elif hasattr(self.tokenizer, 'tokenize'):
            return [self.tokenizer.tokenize(text) for text in itr]
        return super(TweetTokenizer, self).tokenize(itr)


class DictionaryTokenizer(Tokenizer):
    """
    Dictionary tokenizer implementation.
    """

    def __init__(self, vocabulary=None, preprocess=None, separator='_'):
        super(DictionaryTokenizer, self).__init__(None, preprocess, None, vocabulary)
        self.separator = separator
        self.initialize()

    def initialize(self):
        phrases = []
        self.phrases_ = {}
        for w in self.vocabulary:
            temp = [k for k in re.split('(\W)', re.sub(r'_', ' ', w)) if k.strip() != '']
            phrases.append(temp)
            if self.separator.join(temp) not in self.phrases_ or (
                    self.separator.join(temp) in self.phrases_ and len(w) < len(
                self.phrases_[self.separator.join(temp)])):
                self.phrases_[self.separator.join(temp)] = w
        print(self.phrases_)
        self.tokenizer = nltk.MWETokenizer(phrases, separator=self.separator)

    def tokenize(self, itr):
        return [
            [self.phrases_[t] for t in self.tokenizer.tokenize([k for k in re.split('(\W)', text) if k.strip() != ''])
             if t in self.phrases_.keys()] for text in itr]
