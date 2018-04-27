from tklearn.text.en_text import build_vocabulary

s1 = ['this is sentence 1', 'sentence 2 is here :D']
s2 = ['hi this is the second set']

vocab = build_vocabulary(s1, s2)

print(vocab)
