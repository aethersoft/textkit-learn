CAP_TR = {'start:a': 'lower', 'start:A': 'upper', 'lower:a': 'lower', 'lower:A': 'mixed', 'upper:a': 'cap',
          'upper:A': 'upper', 'mixed:a': 'mixed', 'cap:a': 'cap', 'cap:A': 'mixed'}

CAP_F = {'start': 'o', 'lower': 'lower', 'upper': 'o', 'mixed': 'o', 'cap': 'cap'}

SHAPE_MASKS = {'A': 1, 'a': 2, '0': 4, '#': 8}

SHAPE_MASK_STR = {0: '', 1: 'A', 2: 'a', 3: 'Aa', 4: '0', 5: 'A0', 6: 'a0', 7: 'Aa0', 8: '#', 9: 'A#', 10: 'a#',
                  11: 'Aa#', 12: '0#', 13: 'A0#', 14: 'a0#', 15: 'Aa0#'}


def _shape_char(ch):
    if 'a' <= ch <= 'z':
        return 'a'
    elif 'A' <= ch <= 'Z':
        return 'A'
    elif '0' <= ch <= '9':
        return '0'
    else:
        return '#'


def _shape_mask(mask, ch):
    return mask | SHAPE_MASKS[_shape_char(ch)]


def capitalized(wd):
    """
    Returns whether the input is fully capitol or not

    :param wd: input
    :return: state of the input word
    """
    state = 'start'
    for ch in wd:
        if 'a' <= ch <= 'z':
            state = CAP_TR[state + ':a']
        else:
            state = CAP_TR[state + ':A']
    return CAP_F[state]


def word_shape(wd):
    """
    Finds the shape of the word provided
    :param wd: a word to find the shape
    :return:  Shape str
    """
    _mask = 0
    for ch in wd[2:]:
        _mask = _shape_mask(_mask, ch)
    return ''.join(map(_shape_char, wd[:2])) + SHAPE_MASK_STR[_mask]


def char_count(s):
    """
    Calculate count of characters of classes: [A-Z],[a-z],[0-9],[!?],punct
    :param s: input strings
    :return: tuple of  number of (upper, lower, digit, exclamation/question mark, other) letters in text
    """
    u, l, d, e, o = 0, 0, 0, 0, 0
    for ch in s:
        if 'A' <= ch <= 'Z':
            u += 1
        elif 'a' <= ch <= 'z':
            l += 1
        elif '0' <= ch <= '9':
            d += 1
        elif '!' == ch or '?' == ch:
            e += 1
        else:
            o += 1
    return (u, l, d, e, o)


def bigrams(tokens):
    """
    Given a iterable of tokens generated bigrams

    :param tokens: iterable of tokens
    :return: bigrams
    """
    if len(tokens) > 2:
        return [a + "_" + b for a, b in zip(tokens, tokens[1:])]
    else:
        return [a for a in tokens]


def trigrams(tokens):
    """
    Given a iterable of tokens generated trigrams

    :param tokens: iterable of tokens
    :return: bigrams
    """
    if len(tokens) > 3:
        return [a + "_" + b + '_' + c for a, b, c in zip(tokens, tokens[1:], tokens[2:])]
    elif len(tokens) > 2:
        return [a + "_" + b for a, b in zip(tokens, tokens[1:])]
    else:
        return [a for a in tokens]


def make_vocabulary(*args, **kwargs):
    """
    Builds and returns the vocabulary

    :param x: arg list of sentence lists
    :param tokenizer: tokenizer to use: can be string representation of separator or a function with attribute tokenize
    (defaults to tweet tokenizer)
    :return: Vocabulary
    """
    tokenizer = kwargs['tokenizer'] if kwargs['tokenizer'] else lambda x: x.split(' ')
    return [k for t in args for k in tokenizer.tokenize(t)]
