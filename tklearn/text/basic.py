from collections import Counter

__all__ = [
    'build_vocabulary'
]


def build_vocabulary(texts=None, tokenizer=None, preprocess=None, max_vocab=None):
    """
    Builds vocabulary form given text(s) using provided tokenizer. Text pre-processing is performed prior to
    tokenizing.

    :param texts: Input text or list of texts.
    :param tokenizer: None or callable
    :param preprocess: None or callable
    :param max_vocab: Maximum vocabulary size
    :return: Vocabulary set
    """
    if texts is None:
        texts = []
    elif isinstance(texts, str):
        texts = [texts]
    if tokenizer is None:
        def tokenizer(ts):
            return [t.split(' ') for t in ts]
    if preprocess is None:
        def preprocess(ts):
            return ts
    word_freq = Counter()
    tokenize = tokenizer.tokenize if hasattr(tokenizer, 'tokenize') else tokenizer
    for x in tokenize(preprocess(texts)):
        word_freq[x] += 1
    if max_vocab is None:
        frq_words = word_freq.most_common()
    else:
        frq_words = word_freq.most_common(max_vocab - 1)
    vocab = set()
    for x, _ in frq_words:
        vocab.update(x)
    return vocab
