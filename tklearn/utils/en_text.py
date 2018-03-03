cap_tr = {'start:a': 'lower', 'start:A': 'upper', 'lower:a': 'lower', 'lower:A': 'mixed', 'upper:a': 'cap',
          'upper:A': 'upper', 'mixed:a': 'mixed', 'cap:a': 'cap', 'cap:A': 'mixed'}

cap_f = {'start': 'o', 'lower': 'lower', 'upper': 'o', 'mixed': 'o', 'cap': 'cap'}

shape_masks = {'A': 1, 'a': 2, '0': 4, '#': 8}

shape_mask_str = {0: '', 1: 'A', 2: 'a', 3: 'Aa', 4: '0', 5: 'A0', 6: 'a0', 7: 'Aa0', 8: '#', 9: 'A#', 10: 'a#',
                  11: 'Aa#', 12: '0#', 13: 'A0#', 14: 'a0#', 15: 'Aa0#'}

__all__ = [
    'capitalized',
    'shape_char',
    'shape_mask',
    'word_shape',
    'char_count'
]


def capitalized(s):
    state = 'start'
    for ch in s:
        if 'a' <= ch <= 'z':
            state = cap_tr[state + ':a']
        else:
            state = cap_tr[state + ':A']
    return cap_f[state]


def shape_char(ch):
    if 'a' <= ch <= 'z':
        return 'a'
    elif 'A' <= ch <= 'Z':
        return 'A'
    elif '0' <= ch <= '9':
        return '0'
    else:
        return '#'


def shape_mask(mask, ch):
    return mask | shape_masks[shape_char(ch)]


def word_shape(s):
    """
    Finds the shape of the word provided
    :param s: a word to find the shape
    :return:  Shape str
    """
    _mask = 0
    for ch in s[2:]:
        _mask = shape_mask(_mask, ch)
    return ''.join(map(shape_char, s[:2])) + shape_mask_str[_mask]


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
