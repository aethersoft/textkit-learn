from sklearn.base import TransformerMixin, BaseEstimator

__all__ = [
    'TransferVectorizer'
]


class TransferVectorizer(TransformerMixin, BaseEstimator):
    """Collects features from layers of neural network and present as one vector."""

    def __init__(self, model, kwargs=None, layers=-1, pretrained=True):
        """Initialize TransferVectorizer.

        :param model: A trained neural network model or `class` of model to cross_validate.
        :param kwargs: Train parameters.
        :param layers: Layers to extract features from.
        :param pretrained: whether the model is pretrained.
        """
        super(TransferVectorizer, self).__init__()
        self.model = model
        self.kwargs = kwargs
        self.layers = layers
        self.pretrained = pretrained
        self.estimator_ = None

    def fit(self, X, y=None):
        """Fits TransferVectorizer. Trains the model is not pretrained.

        :param X: Training features.
        :param y: Labels.
        :return: self.
        """
        if self.kwargs is None:
            self.estimator_ = self.model
        else:
            self.estimator_ = self.model(**self.kwargs)
        if not self.pretrained:
            self.estimator_.fit(X, y)
        # freeze the layers
        for param in self.estimator_.model.parameters():
            param.requires_grad = False
        self.estimator_.model.return_layers = self.layers
        return self

    def transform(self, X):
        """Transform input in to features using the model provided in init.

        :param X: Raw input features.
        :return: Extracted features.
        """
        features = self.estimator_.predict_proba(X)
        return features
