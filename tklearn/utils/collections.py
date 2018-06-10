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


def collect(func, ipt, out, *args, **kwargs):
    """
    Collects output for each input item and add them to provided list/set as set in output parameter.

    :param func: a function to apply
    :param ipt: input to function
    :param out: collect outputs here should support += operator or should be a set
    :param args: args to func call
    :param kwargs: args to func call
    :return:
    """
    for s in ipt:
        p = func(s, *args, **kwargs)
        if isinstance(out, set):
            out.update(p)
        else:
            out += list(p)
    return out


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


def parallel_sort(*args):
    return [list(t) for t in zip(*sorted(zip(*args), key=lambda x: x[0]))]
