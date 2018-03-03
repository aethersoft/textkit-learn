# Tweet Tokenizer usage example
from tklearn.preprocessing import TweetTokenizer

X = ['this is a sample text', 'this is the second text']

tt = TweetTokenizer()
tokens = tt.fit_transform(X)

print(tokens)
