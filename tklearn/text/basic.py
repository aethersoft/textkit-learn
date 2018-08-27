__all__ = [
    'build_vocabulary'
]


def build_vocabulary(texts=None, tokenizer=None, preprocess=None):
    """
    Builds vocabulary form given text(s) using provided tokenizer. Text pre-processing is performed prior to
    tokenizing.

    :param texts: Input text or list of texts.
    :param tokenizer: None or callable
    :param preprocess: None or callable
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
    vocab = set()
    for x in tokenizer(preprocess(texts)):
        vocab.update(x)
    return vocab
