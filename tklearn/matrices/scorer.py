from typing import Text, Iterable, Any

from sklearn.metrics import get_scorer


class ScoreFunction:
    def __init__(self, scorer: Text):
        """ Initialize score function

        Parameters
        ----------
        scorer : Text
            Scikit-learn Name of scorer.
        """
        self.scorer = get_scorer(scorer)

    # noinspection PyProtectedMember
    def __call__(self, y_true: Iterable, y_pred: Iterable) -> Any:
        """ Score function caller.

        Parameters
        ----------
        y_true
            Ground Truth

        y_pred
            True Predictions

        Returns
        -------
            Score Function
        """
        return self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)


# noinspection PyProtectedMember
def get_score_func(scorer: Text) -> ScoreFunction:
    """ Get score function from the given string or callable.

    Parameters
    ----------
    scorer
        Name of the scorer as defined in scikit-learn.

    Returns
    -------
    score_function
        Score function.
    """
    return ScoreFunction(scorer)
