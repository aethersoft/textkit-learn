import json
from os import walk, environ
from os.path import join


def download():
    raise NotImplementedError('Download function has not been implemented yet.')


def resource_path(*args):
    """
    Gets resource folder name from environment path.
    :param args: resource path if exists
    :return: Returns path to the resource folder at the root
    """
    try:
        return join(environ['TKLEARN_RESOURCES'], *args)
    except KeyError:
        msg = 'The environment variable \'TKLEARN_RESOURCES\' is not set.'
        raise LookupError(msg)


def get_lexicon(name):
    """
    Gets resources from resource path.
     Resource path should contain json file indicating the resources and how to access them.
    :param name: name of the lexicon resource
    :return: path to lexicon
    """
    resources = json.load(open(resource_path('resources.json')))
    lexicons = resources['lexicons']
    path = resource_path(*lexicons[name]['path'])
    return path


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
