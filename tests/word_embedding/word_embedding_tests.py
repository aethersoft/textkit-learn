import unittest

from tklearn.text.embedding import load_embedding, load_word2vec


class WordEmbeddingTestCase(unittest.TestCase):
    def test_word_embedding_vocab(self):
        word2vec_path = 'D:\Documents\Resources\Models\word2vec_twitter_model\word2vec_twitter_model.bin'
        word2vec_twitter = load_word2vec(word2vec_path)
        assert word2vec_twitter.vocabulary is not None

    def test_load_embedding(self):
        embedding_path = 'D:\\Documents\\Resources\\Models\\textkit-resources\\embeddings\\EdinburgheEmbeddings\\w2v.twitter.edinburgh.100d.csv.gz'
        word2vec_edinburgh = load_embedding(embedding_path)
        print(word2vec_edinburgh['i'])
        assert word2vec_edinburgh['i'] is not None

    def test_load_two_embedding(self):
        embedding_path1 = 'D:\Documents\Resources\Models\word2vec_google_model\GoogleNews-vectors-negative300.bin'
        embedding_path2 = 'D:\Documents\Resources\Models\\textkit-resources\embeddings\Emoji2Vec\emoji2vec.txt.gz'
        word2vec_google = load_word2vec(embedding_path1)
        vocab_len1 = len(word2vec_google.vocabulary)
        word2vec_edinburgh = load_embedding(embedding_path2)
        vocab_len2 = len(word2vec_edinburgh.vocabulary)
        word2vec_google.append(word2vec_edinburgh)
        print('Asserting: {}+{}=={}'.format(vocab_len1, vocab_len2, word2vec_google))
        assert vocab_len1 + vocab_len2 == len(word2vec_google.vocabulary)

    def test_load_fasttext(self):
        embedding_path = 'D:\Documents\Resources\Models\\textkit-resources\embeddings\FastText\wiki-news-300d-1M.vec.gz'
        wiki_fasttext = load_embedding(embedding_path, word_first=True, leave_head=True)
        assert wiki_fasttext['i'] is not None


if __name__ == '__main__':
    unittest.main()
