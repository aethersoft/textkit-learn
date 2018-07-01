import unittest

from tklearn.preprocessing import DictionaryTokenizer


class DictionaryTokenizerTestCase(unittest.TestCase):
    def test_tokenize(self):
        vocabulary = ['this', 'good', 'example', ':)', ':(', 'Sri_Lanka']
        tok = DictionaryTokenizer(vocabulary)
        text = 'this... is a good example :):D:( Love Sri Lanka'
        tokens = tok.tokenize([text])
        assert len(tokens[0]) == 11


if __name__ == '__main__':
    unittest.main()
