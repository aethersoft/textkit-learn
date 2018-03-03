import sys

from gensim.models import KeyedVectors

from tklearn.preprocessing import TweetTokenizer, EmbeddingExtractor

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
print('-' * 25)
print('Shape of Extracted Embedding Matrix: {}'.format(X['embedding_matrix'].shape))
print('-' * 25 + 'Word index' + '-' * 25)
print(X['word_index'])
print('-' * 25 + 'Embedding Matrix' + '-' * 25)
print(X['embedding_matrix'])
sys.exit(0)
