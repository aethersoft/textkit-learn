from tklearn.pipeline import FeatureConcatenate
from tklearn.preprocessing import TweetTokenizer

fc = FeatureConcatenate([('a', TweetTokenizer()), ('b', TweetTokenizer(vocab=['1', '2']))])

X = ['test 1', 'test 2']

t = fc.fit_transform(X, None)

print(t)
