"""
This example reads texts and return the average glove embeddings for sentence.

 >>> get_features(
 >>>    tweet_samples,
 >>>    embedding=WordEmbedding(model),
 >>>    preprocessor=TweetPreprocessor(normalize=['link', 'mention']),
 >>>    tokenizer=TweetTokenizer()
 >>> ).shape
 >>> (5, 100)
"""

from typing import List, Text

import gensim.downloader as api
import pandas as pd
from nltk import TweetTokenizer

from tklearn.embedding import WordEmbedding
from tklearn.feature_extraction import make_embedding_transformer
from tklearn.preprocessing import TweetPreprocessor, TextPreprocessor

model = api.load('glove-twitter-100')


def get_features(texts: List[Text], embedding: WordEmbedding, preprocessor: TextPreprocessor, tokenizer):
    pp, tk = preprocessor, tokenizer
    tokenized_texts = pd.Series(texts) \
        .apply(pp.preprocess) \
        .apply(tk.tokenize)
    et = make_embedding_transformer(embedding)  # is a function transformer so not required to return that
    return et.fit_transform(tokenized_texts)


if __name__ == '__main__':
    tweet_samples = [
        'I wrote this haiku because Twitter has line breaks no other reason',
        'I had a GREAT week, thanks to YOU! If you need anything, please reach out. ❤️ ❤️ ❤️ '
        '#WorldSmileDay pic.twitter.com/ZpVmQPmcyc',
        'My heart goes out to the Malaysian people. This is such a tragedy. Words can\'t express how sad it is. '
        'I wish we could just have peace. #MH17',
        'This is what I\'m in the mood for right now. :) #nom https://t.co/WOE7VAPgci',
        '''My #SocialMedia seniors book in 1200+ stores in Canada, next to #1 best seller by Harper Lee. 
        TY @ShopprsDrugMart pic.twitter.com/gL6WfAVQM1'''
    ]
    X = get_features(
        tweet_samples,
        embedding=WordEmbedding(model),
        preprocessor=TweetPreprocessor(normalize=['links', 'mentions']),
        tokenizer=TweetTokenizer()
    )
    print(X.shape)
    assert X.shape == (5, 100)
