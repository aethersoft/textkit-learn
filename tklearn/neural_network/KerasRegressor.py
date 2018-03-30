from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, RegressorMixin


class KerasRegressor(ABC, BaseEstimator, RegressorMixin):
    """
    An more abstract version of KerasRegressor for scikit-learn with custom preprocess pipeline.
    """

    def __init__(self, batch_size=16, epochs=8):
        """
        Initialize the Regressor.

        :param batch_size: Number of samples per gradient update. If unspecified, it will default to 16
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        """
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data

        :param X: Training vector with  shape = [n_samples, n_features],
         where n_samples in the number of samples and  n_features is the number of features.
        :param y: Target vector relative to X of shape shape = [n_samples]
        :return: self
        """
        X, y = self.preprocess(X, y)
        self.model_ = self.build_model(X, y)
        self.model_.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X / Predict target values of X given a model

        :param X: Testing vector
        :return: Predicted values
        """
        X, _ = self.preprocess(X)
        return self.model_.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy on the given test data and labels.

        :param X: Test samples
        :param y: True labels for X
        :param sample_weight: Sample weights
        :return:
        """
        _, acc = self.evaluate(X, y, sample_weight)
        return acc

    def evaluate(self, x_train, y, sample_weight=None):
        """
        Returns the (loss, accuracy) tuple on the given test data and labels.

        :param x_train: Test samples with  shape = [n_samples, n_features],
         where n_samples in the number of samples and  n_features is the number of features.
        :param y: True labels for X of shape shape = [n_samples]
        :param sample_weight: Sample weights.
        :return:
        """
        x_train = self.preprocess(x_train)
        return self.model_.evaluate(x_train, y, batch_size=self.batch_size)

    @abstractmethod
    def build_model(self, X, y):
        """
        Compiles the CNN model

        :return: compiled model
        """
        pass

    @abstractmethod
    def preprocess(self, X, y=None):
        """
        Preprocess the data and return the preprocessed input for the model

        :return: model input
        """
        return X, y
