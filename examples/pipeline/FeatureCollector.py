from tklearn.pipeline import FeatureList
from tklearn.preprocessing import TweetTokenizer

fc = FeatureList([('a', TweetTokenizer()), ('b', TweetTokenizer(vocab=['1', '2']))])

X = ['test 1', 'test 2']

t = fc.fit_transform(X, None)

print(t)
