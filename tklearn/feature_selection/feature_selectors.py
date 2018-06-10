from sklearn.preprocessing import FunctionTransformer

__all__ = [
    'SelectByIndex',
    'SelectByKey',
]


class SelectByIndex(FunctionTransformer):
    def __init__(self, index, axis=1):
        """
        Given index selects the feature(s) indicated by that index. The input to the fit function should support __getitem__.

        :param index: a index to select
        :type axis: axis
        """
        super(SelectByIndex, self).__init__(self._select_by_getitem_func, validate=False)
        self.index = index
        self.axis = axis

    def _select_by_getitem_func(self, X):
        assert hasattr(X, '__getitem__'), '\'{}\' object is not subscriptable.'.format(type(X))
        assert 0 < self.axis < 3, 'SelectByIndex supports only axis 1 or 2'
        if self.axis == 2:
            return list(zip(*X))[self.index]
        return X[self.index]


class SelectByKey(SelectByIndex):
    def __init__(self, key):
        """
        Given a key selects the feature(s) indicated by that key. The input to the fit function should support __getitem__.

        :param key: a key to select
        """
        super(SelectByKey, self).__init__(key, 1)
