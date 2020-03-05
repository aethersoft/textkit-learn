from os import environ
from os.path import join, expanduser

configs = dict(
    RESOURCE_PATH=expanduser(environ.get('OLANG_PATH', join('~', '.olang'))),
)
