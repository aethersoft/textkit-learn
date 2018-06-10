import requests


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
