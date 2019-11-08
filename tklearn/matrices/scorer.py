from typing import Text, Iterable, Any

from sklearn.metrics import get_scorer


class ScoreFunction:
    def __init__(self, scorer: Text):
        """ Initialize score function

        :param scorer: Text
            Scikit-learn Name of scorer.
        """
        self.scorer = get_scorer(scorer)

    # noinspection PyProtectedMember
    def __call__(self, y_true: Iterable, y_pred: Iterable) -> Any:
        """ Score function caller.

        :param y_true:
            Ground Truth
        :param y_pred:
            True Predictions
        :return: Score Function
        """
        return self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)


# noinspection PyProtectedMember
def get_score_func(scorer: Text) -> ScoreFunction:
    """Get score function from the given string or callable.

    :param scorer: Name of the scorer as defined in scikit-learn.
    :return: Score function.
    """
    return ScoreFunction(scorer)
