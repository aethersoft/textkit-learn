from abc import ABCMeta, abstractmethod
from typing import Text


class TextPreprocessor(metaclass=ABCMeta):
    """All preprocessors should inherit this class for compatibility with `olang` module.
    """

    @abstractmethod
    def preprocess(self, s: Text) -> Text:
        """Function should take a `str` type input and return output of `str` type.

        It is expected that the input to be transformed according to a specific use case.
        :param s: Input text of type `str`.
        :return: Preprocessed text of input `s`.
        """
        raise NotImplementedError
