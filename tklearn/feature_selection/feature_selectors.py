from sklearn.preprocessing import FunctionTransformer


class SelectByIndex(FunctionTransformer):
    def __init__(self, index):
        """
        Given index selects the feature(s) indicated by that index. The input to the fit function should support __getitem__.

        :param index: a index to select
        """
        super(SelectByIndex, self).__init__(self._select_by_getitem_func, validate=False, kw_args={'item': index})

    @staticmethod
    def _select_by_getitem_func(X, item):
        assert hasattr(X, '__getitem__'), '\'{}\' object is not subscriptable.'.format(type(X))
        return X[item]


class SelectByKey(SelectByIndex):
    def __init__(self, key):
        """
        Given a key selects the feature(s) indicated by that key. The input to the fit function should support __getitem__.

        :param key: a key to select
        """
        super(SelectByKey, self).__init__(key)
