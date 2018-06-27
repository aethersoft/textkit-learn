import unittest

import numpy as np

from tklearn.feature_extraction import AffectiveTweetsVectorizer
from tklearn.preprocessing import DictionaryTokenizer

class LexiconFeaturizerTestCase(unittest.TestCase):
    def test_affinn(self):
        lexicon = 'affinn'
        l = AffectiveTweetsVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert 3.0 == res[0][0], 'Invalid implementation of {}'.format(lexicon)

    def test_senti_streangth(self):
        lexicon = 'senti_streangth'
        l = AffectiveTweetsVectorizer(lexicon)
        text = 'I love you but hate the current political climate.'
        res = l.fit_transform([text])
        assert 3.0 == res[0][0], 'Invalid implementation of {}'.format(lexicon)
        assert -4.0 == res[0][1], 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_hashtag_score(self):
        lexicon = 'nrc_hashtag_score'
        l = AffectiveTweetsVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert 0.0 != res[0][0], 'Invalid implementation of {}'.format(lexicon)
        assert 0.0 != res[0][1], 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_hashtag(self):
        lexicon = 'nrc_hashtag'
        l = AffectiveTweetsVectorizer(lexicon)
        text = '#good job'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_exp_emotion(self):
        lexicon = 'nrc_exp_emotion'
        l = AffectiveTweetsVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_emotion(self):
        lexicon = 'nrc_emotion'
        l = AffectiveTweetsVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_nrc_affect_int(self):
        lexicon = 'nrc_affect_int'
        l = AffectiveTweetsVectorizer(lexicon)
        text = 'This is utterly excellent!'
        res = l.fit_transform([text])
        assert np.sum(res) > 0, 'Invalid implementation of {}'.format(lexicon)

    def test_neg(self):
        lexicon = 'neg'
        l = AffectiveTweetsVectorizer(lexicon)
        text_pos = 'this is a good result'
        text_neg = 'this is not a good result'
        res = l.fit_transform([text_pos, text_neg])
        assert res[0][0] == 0, 'Invalid implementation of {}'.format(lexicon)
        assert res[1][0] == 1, 'Invalid implementation of {}'.format(lexicon)

    def test_edinburgh(self):
        lexicon = 'edinburgh'
        l = AffectiveTweetsVectorizer([lexicon])
        text_pos = 'this is a good result'
        res = l.fit_transform([text_pos])
        assert res[0][0] != 0, 'Invalid implementation of {}'.format(lexicon)

    def test_emoji(self):
        lexicon = 'emoji'
        l = AffectiveTweetsVectorizer(lexicon)
        res = l.fit_transform(['👽💬'])
        assert res[0][0] != 0, 'Invalid implementation of {}'.format(lexicon)

    def test_liwc(self):
        lexicon = 'liwc'
        l = AffectiveTweetsVectorizer(lexicon)
        text_pos = 'this is a good result'
        res = l.fit_transform([text_pos])
        print(res)
        assert res[0][0] != 0, 'Invalid implementation of {}'.format(lexicon)


if __name__ == '__main__':
    unittest.main()
