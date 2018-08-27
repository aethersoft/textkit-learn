from sklearn.base import TransformerMixin, BaseEstimator

from tklearn.utils import load_pipeline

__all__ = ['TransferFeaturizer']


class TransferFeaturizer(BaseEstimator, TransformerMixin):
    """
    Transfer learning with pickle support. Doesn't save the model but loads when initialized.
    """

    def __init__(self, model_path, layer_name=None):
        self.model_path = model_path
        self.layer_name = layer_name
        self.initialize()

    def initialize(self):
        emo_clf = load_pipeline(self.model_path)
        emo_clf.steps[-1][-1].transfer(True, self.layer_name)
        self._emo_clf = emo_clf

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        if hasattr(self, '_emo_clf'):
            return self._emo_clf.predict(X)
        else:
            return [[] for _ in X]

    def __getstate__(self):
        return {'model_path': self.model_path, 'layer_name': self.layer_name}

    def __setstate__(self, state):
        self.model_path = state['model_path']
        self.layer_name = state['layer_name']
        self.initialize()
