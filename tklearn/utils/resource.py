from os import walk
from os.path import join, dirname


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
    for folder, subs, files in walk(base_path):
        for filename in files:
            if predicate(join(folder, filename)):
                yield (join(folder, filename))
