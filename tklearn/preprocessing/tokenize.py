import re

import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.twitter import CMUTweetTagger

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
        if isinstance(self.tokenizer, CMUTweetTagger):
            tag_set = self.tokenizer.tag(itr)
            return [list(list(zip(*tags))[0]) for tags in tag_set]
        elif hasattr(self.tokenizer, 'tokenize'):
            return [self.tokenizer.tokenize(text) for text in itr]
        return super(TweetTokenizer, self).tokenize(itr)


class DictionaryTokenizer(Tokenizer):
    """
    Dictionary tokenizer implementation.
    """

    def __init__(self, vocabulary=None, preprocess=None, separator='_', ignore_vocab=True):
        super(DictionaryTokenizer, self).__init__(None, preprocess, None, vocabulary)
        self.separator = separator
        self.ignore_vocab = ignore_vocab
        self.initialize()

    def initialize(self):
        st = nltk.TweetTokenizer()
        self.phrases_ = []
        self.phrase2vocab_ = {}
        for w in self.vocabulary:
            temp = [k for k in st.tokenize(re.sub(r'_', ' ', w)) if k.strip() != '']
            self.phrases_.append(temp)
            if self.separator.join(temp) not in self.phrase2vocab_ or (
                    self.separator.join(temp) in self.phrase2vocab_ and len(w) < len(
                self.phrase2vocab_[self.separator.join(temp)])):
                self.phrase2vocab_[self.separator.join(temp)] = w
        self.vocabulary = None

    def tokenize(self, itr):
        st = nltk.TweetTokenizer()
        tokenizer = nltk.MWETokenizer(self.phrases_, separator=self.separator)
        return [
            [self.phrase2vocab_[t] if t in self.phrase2vocab_ else t for t in
             tokenizer.tokenize([k for k in st.tokenize(text) if k.strip() != ''])
             if t in self.phrase2vocab_.keys() or self.ignore_vocab] for text in itr]
