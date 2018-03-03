import sys

from gensim.models import KeyedVectors

from tklearn.preprocessing import TweetTokenizer, EmbeddingExtractor, EmbeddingAverager

WORD2VEC_PATH = 'D:\Documents\Resources\Models\word2vec_twitter_model\word2vec_twitter_model.bin'
print('Starting testing Embedding Averager...')
print('Loading Word2Vec...', end=' ')
wv = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True, unicode_errors='ignore')
print('[Done]')
X = ['this is a sample text', 'this is the second text']
tok = TweetTokenizer()
X = tok.fit_transform(X)
ee = EmbeddingExtractor(wv)
X = ee.fit_transform(X)
ea = EmbeddingAverager()
X = ea.fit_transform(X)
print('-' * 25)
print('Shape of Mean Embedding Vector: {}'.format(X.shape))
sys.exit(0)
