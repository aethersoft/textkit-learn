import json


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
