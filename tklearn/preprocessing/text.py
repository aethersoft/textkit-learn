from abc import ABCMeta
from abc import abstractmethod
from typing import Text

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'TextPreprocessor',
]


class TextPreprocessor(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """ All preprocessors should inherit this class for compatibility with `olang` module."""

    @abstractmethod
    def preprocess(self, s: Text) -> Text:
        """ Function should take a `str` type input and return output of `str` type.

        It is expected that the input to be transformed according to a specific use case.

        Parameters
        ----------
        s
            Input text of type `str`.

        Returns
        -------
            Preprocessed text of input `s`.
        """
        raise NotImplementedError

    def fit(self, X, y=None) -> 'TextPreprocessor':
        """ Do nothing and return the estimator unchanged

        Parameters
        ----------
        X
            array-like
        y
            Nothing

        Returns
        -------
            Unchanged estimator
        """
        return self

    def transform(self, X) -> list:
        """ Preprocess each element of X

        Parameters
        ----------
        X
            {array-like}, shape [n_samples]

        Returns
        -------
            preprocessed text
        """
        return [self.preprocess(x) for x in X]
