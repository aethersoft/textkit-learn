import unittest

from tklearn.feature_extraction import EmbeddingTransformer
from tklearn.preprocessing import TweetTokenizer
from tklearn.text import build_vocabulary, get_weights


class EmbeddingTransformerTestCase(unittest.TestCase):
    def test_embedding_extraction(self):
        train_text = ['this is random sample 2']
        test_text = ['this is random sample']
        corpus = train_text + test_text
        tokenizer = TweetTokenizer().tokenize
        vocab = build_vocabulary(corpus, tokenizer=tokenizer, preprocess=None)
        weight_mat, word_idx = get_weights(vocabulary=vocab, default='random')
        transformer = EmbeddingTransformer(word_index=word_idx, tokenizer=tokenizer, preprocess=None)
        train_seq = transformer.fit_transform(train_text)
        test_seq = transformer.transform(test_text)
        print(train_seq, test_seq)
        self.assertEqual(len(train_seq), len(test_seq))


if __name__ == '__main__':
    unittest.main()
