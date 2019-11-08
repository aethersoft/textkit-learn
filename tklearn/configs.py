from os import environ
from os.path import join, expanduser

configs = dict(
    OLANG_PATH=expanduser(environ.get('OLANG_PATH', join('~', '.olang'))),
)
