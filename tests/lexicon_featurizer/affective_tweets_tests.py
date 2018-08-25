import unittest

import numpy as np

from tklearn.feature_extraction import LexiconVectorizer


class LexiconFeaturizerTestCase(unittest.TestCase):
    def test_affinn(self):
        lexicon = 'AFINN'
        l = LexiconVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert 3.0 == res[0][0], 'Invalid implementation of {}'.format(lexicon)

    def test_senti_streangth(self):
        lexicon = 'SentiStrength'
        l = LexiconVectorizer(lexicon)
        text = 'I love you but hate the current political climate.'
        res = l.fit_transform([text])
        assert 3.0 == res[0][0], 'Invalid implementation of {}'.format(lexicon)
        assert -4.0 == res[0][1], 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_hashtag_score(self):
        lexicon = 'NRCHS'
        l = LexiconVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert 0.0 != res[0][0], 'Invalid implementation of {}'.format(lexicon)
        assert 0.0 != res[0][1], 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_hashtag(self):
        lexicon = 'NRCHEA'
        l = LexiconVectorizer(lexicon)
        text = '#good job'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_exp_emotion(self):
        lexicon = 'NRC10E'
        l = LexiconVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_emoticon(self):
        lexicon = 'NRCWEA'
        l = LexiconVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_afinn_emoticon(self):
        lexicon = 'emoticon'
        l = LexiconVectorizer(lexicon)
        text = 'This is utterly excellent! :)'
        res = l.fit_transform([text])
        assert np.sum(res) == 2.0, 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_affect_int(self):
        lexicon = 'NRCAI'
        l = LexiconVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_neg(self):
        lexicon = 'Negations'
        l = LexiconVectorizer(lexicon)
        text_pos = 'this is a good result'
        text_neg = 'this is not a good result'
        res = l.fit_transform([text_pos, text_neg])
        assert res[0][0] == 0, 'Invalid implementation of {}'.format(lexicon)
        assert res[1][0] == 1, 'Invalid implementation of {}'.format(lexicon)

    def test_liwc(self):
        lexicon = 'LIWC'
        l = LexiconVectorizer(lexicon)
        text_pos = 'this is a good result'
        res = l.fit_transform([text_pos])
        assert res[0][0] != 0, 'Invalid implementation of {}'.format(lexicon)


if __name__ == '__main__':
    unittest.main()
