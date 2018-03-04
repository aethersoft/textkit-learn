from os import walk, environ
from os.path import join


def download():
    raise NotImplementedError('Download function has not been implemented yet.')


def resource_path():
    """
    Gets resource folder name from environment path.
    :return: Returns path to the resource folder at the root
    """
    try:
        return environ['TKLEARN_RESOURCES']
    except KeyError:
        msg = 'The environment variable \'TKLEARN_RESOURCES\' is not set.'
        raise LookupError(msg)


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
