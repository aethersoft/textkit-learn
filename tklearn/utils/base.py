import json

import requests
from sklearn.externals import joblib


class DiskCache:
    def __init__(self, path, auto_flush=False):
        """
        constructs cache object related with file at path

        :param path: path to cache file
        """
        self.path = path
        self.auto_flush = auto_flush
        self.cache = dict()
        self.has_changed = False
        self.initialize()

    def initialize(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as fp:
                self.cache.update(json.load(fp))
        except IOError:
            with open(self.path, 'w', encoding='utf-8') as fp:
                json.dump(self.cache, fp)

    def has(self, item):
        return item in self.cache

    def __getitem__(self, item):
        return self.cache[item]

    def __setitem__(self, key, value):
        self.has_changed = True
        self.cache[key] = value
        if self.auto_flush:
            self.flush()

    def flush(self):
        if self.has_changed:
            with open(self.path, 'w', encoding='utf-8') as fp:
                json.dump(self.cache, fp)


class WebService:
    def __init__(self, ip, port, use_https=False):
        self.ip = ip
        self.port = port
        self.use_https = use_https

    def post(self, *args, **kwargs):
        if self.use_https:
            addr = 'https://{}:{}/'.format(self.ip, self.port)
        else:
            addr = 'http://{}:{}/'.format(self.ip, self.port)
        addr += '/'.join(args)
        r = requests.post(addr, data=kwargs)
        if r.status_code == 200:
            return eval(r.text)
        else:
            msg = \
                'Server Error with status code {}. Please check server log for more information.'.format(r.status_code)
            raise Exception(msg)

    def __getattr__(self, item):
        def f(**kwargs):
            return self.post(item, **kwargs)

        return f


def load_pipeline(path):
    """
    Loads sci-kit learn pipeline from persistent storage.

    :param path: Path to the sci-kit learn pipeline.
    :return: pipeline to save
    """
    pipeline = joblib.load('{}.pkl'.format(path))
    if hasattr(pipeline.steps[-1][-1], 'load'):
        pipeline.steps[-1][-1].load(path)
    return pipeline


def save_pipeline(path, pipeline):
    """
    Saves sci-kit learn pipeline to persistent storage

    :param path: Path to the sci-kit learn pipeline.
    :param pipeline: pipeline to save
    :return:
    """
    joblib.dump(pipeline, '{}.pkl'.format(path))
    if hasattr(pipeline.steps[-1][-1], 'save'):
        pipeline.steps[-1][-1].save(path)
