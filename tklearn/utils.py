import logging as logging
import os
import shutil
import sys
import time
import zipfile
from logging import config as logging_conf
from urllib import request

import pandas as pd

__all__ = [
    'pprint', 'get_logger', 'download',
    'dict_normalize', 'dict2string'
]


def dict_normalize(d):
    df = pd.json_normalize(d, sep='.')
    return df.to_dict(orient='records')[0]


def dict2string(d):
    output = ''
    for k, v in d.items():
        output += '{:<15}{:>15}\n'.format(k, v)
    return output


def pprint(obj, **kwargs):
    """ Pretty print the provided object.

    Parameters
    ----------
    obj
        Object to print.
    kwargs

    Returns
    -------
        None
    """
    flush = kwargs['flush'] if 'flush' in kwargs else True
    if type(obj) == pd.DataFrame:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(obj, flush=flush)
    elif 'type' in kwargs and kwargs['type'] == 'table' and type(obj) == dict:
        obj = pd.DataFrame(obj)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(obj, flush=flush)
    elif 'type' in kwargs and kwargs['type'] is not None:
        print('[pprint]\tWarning: Invalid type parameter %s.' % kwargs['type'])
        print(obj, flush=flush)
    else:
        print(obj, flush=flush)


def get_logger(name, config=None):
    if config is None:
        config = {
            'version': 1,
            'formatters': {
                'default_formatter': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
            },
            'handlers': {
                'h': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default_formatter',
                    'level': logging.DEBUG
                }
            },
            'root': {
                'handlers': ['h'],
                'level': logging.DEBUG,
            }
        }
    logging_conf.dictConfig(config)
    return logging.getLogger(name)


class ReportHook:
    def __init__(self):
        self.start_time = None

    def __call__(self, count, block_size, total_size):
        if count == 0 or self.start_time is None:
            self.start_time = time.time()
            return
        duration = max(time.time() - self.start_time, 1)
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r- %d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()


def download(url, root, filename, unzip=False):
    """ Download file from a provided URL to '{root}/{filename}'.

    Parameters
    ----------
    url
        URL to download the file from.
    root
        Root folder to save the file.
    filename
        Name of the file to be saved.
    unzip
        Whether to unzip the file.

    Returns
    -------
        Nothing
    """
    if unzip:
        file = os.path.join(root, os.path.splitext(filename)[0])
    else:
        file = os.path.join(root, filename)
    if os.path.exists(file):
        if input('\n- File with the same name exist. '
                 'Do you want to delete that and download the files again? [y/N] ').lower() == 'y':
            shutil.rmtree(root, ignore_errors=True)
        else:
            return
    if not os.path.exists(root):
        os.makedirs(root)
    file = os.path.join(root, filename)
    print(f'Download Progress\t[{filename}]\t:', end='\t')
    request.urlretrieve(url, file, ReportHook())
    print()
    if unzip:
        with zipfile.ZipFile(file, 'r') as zf:
            zf.extractall(root)
        os.remove(file)
