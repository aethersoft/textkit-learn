import os
from os.path import join, dirname

import numpy as np
import scipy.sparse as sp


def contains_iterable(task):
    """
    Checks whether task has iterable elements
    :param task: task to watch
    :return: whether task has iterable objects as its elements
    """
    if len(task) == 0:
        return False
    return hasattr(task[0], '__iter__') and not isinstance(task[0], str)


def concatenate_array(arr) -> np.array:
    """
    Concatenates given arrays
    :param arr: arrays to concatenate
    :return: concatenated arrays
    """
    assert arr is not None, 'Unable to concatenate an array with None type.'
    if len(arr) > 0:
        if sp.issparse(arr[0]):
            return sp.vstack(arr, format=arr[0].format)
        else:
            return np.concatenate(arr)
    return arr


def invert_array(arr):
    """
    Inverts an array and returns the new inverted array
    :param arr: an array to invert
    :return: inverted array
    """
    assert arr is not None, 'Unable to invert an array with None type.'
    inv_arr = np.empty(len(arr), dtype=int)
    inv_arr[arr] = np.arange(len(arr))
    return inv_arr


def apply(func, arr, *args, **kwargs):
    """
    Appplies given function on given array and returns the result as an array
    :param func: a function to apply
    :param arr: array the function is applied on
    :param args: args to the function
    :param kwargs: args to function dict
    :return:
    """
    return [func(item, *args, **kwargs) for item in arr]


def resource_path():
    """
    Gets resource folder.
    :return: Returns path to the resource folder at the root
    """
    return join(dirname(__file__), '..', '..', 'resources')


def list_files(base_path, predicate):
    """
    Generator for walking through the folder structure

    :param base_path:
    :param predicate:
    :return:
    """
    for folder, subs, files in os.walk(base_path):
        for filename in files:
            if predicate(os.path.join(folder, filename)):
                yield (os.path.join(folder, filename))


def get_bigrams(tokens):
    """
    Given a iterable of tokens generated bigrams

    :param tokens: iterable of tokens
    :return: bigrams
    """
    return [a + " " + b for a, b in zip(tokens, tokens[1:])]


def merge_dicts(*dicts):
    """
    Given number of dicts, merge them into a new dict as a shallow copy.
    :param dicts: dicts to merge
    :return: merge of all dicts
    """
    if len(dicts) == 0:
        return dict()
    z = dicts[0].copy()
    for i in range(1, len(dicts)):
        z.update(dicts[i])
    return z
