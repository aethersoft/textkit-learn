from tklearn.utils.en_text import build_vocabulary

list_s = [['this is sentence 1', 'sentence 2 is here :D'], ['hi this is the second set']]

vocab = build_vocabulary(list_s)

print(vocab)
