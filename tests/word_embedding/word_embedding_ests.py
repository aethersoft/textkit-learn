import unittest

from tklearn.text.embedding import load_word2vec


class WordEmbeddingTestCase(unittest.TestCase):
    def test_word_embedding_vocab(self):
        word2vec_path = 'D:\Documents\Resources\Models\word2vec_twitter_model\word2vec_twitter_model.bin'
        word2vec_twitter = load_word2vec(word2vec_path)
        assert word2vec_twitter.vocabulary is not None


if __name__ == '__main__':
    unittest.main()
