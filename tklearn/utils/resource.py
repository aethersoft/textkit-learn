import os
from os import walk, environ, remove
from os.path import join
from urllib.request import urlretrieve
from zipfile import ZipFile

RESOURCE_URL = 'https://github.com/aethersoft/textkit-learn/releases/download/v0.1/textkit-resources.zip'


def download(download_dir):
    """
    Downloads resources and saves in the default path.
    :param download_dir: path to custom resource folder location.
                 If custom location is use environment variable 'TKLEARN_RESOURCES' to point to that location.
    :return: Nothing
    """
    extract_path = join(download_dir, '..')
    download_path = join(extract_path, 'temp.zip')
    print('Downloading files...', end='')
    urlretrieve(RESOURCE_URL, download_path)
    print('Done')
    print('Extracting files...', end='')
    with ZipFile(download_path) as zip:
        zip.extractall(extract_path)
    try:
        os.rename(join(extract_path, zip.namelist()[0]), download_dir)
        print('Done')
    except OSError:
        print('Failed')
    print('Removing temporary files...', end='')
    remove(download_path)
    print('Done')


def resource_path(*args):
    """
    Gets resource folder name from environment path.
    :param args: resource path if exists
    :return: Returns path to the resource folder at the root
    """
    try:
        path = environ['TKLEARN_RESOURCES']
    except KeyError:
        msg = 'The environment variable \'TKLEARN_RESOURCES\' is not set.'
        raise LookupError(msg)
    return join(path, *args)


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
