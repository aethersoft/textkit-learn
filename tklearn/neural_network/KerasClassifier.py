from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin


class KerasClassifier(ABC, BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Initialize the classifier
        """
        pass

    def fit(self, Xs, y):
        """
        Fits the training data to the defined classifier

        :param X: Features
        :param y: Labels
        :return: self
        """
        raise NotImplementedError

    def predict(self, X, y=None, *args, **kwargs):
        """
        Predicts the classes and return them
        :param X: Feature matrix [n_samples, num_features]
        :param y: None
        :param args: Extra args
        :param kwargs: Extra kwargs
        :return: predicted classes
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath):
        """
        Saves the model weights in a file provided by path
        :type filepath: path to the file
        :return: status
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath):
        """
        Loads model from the filepath
        :type filepath: path to the file
        :return: Saved model
        """
        raise NotImplementedError

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass
