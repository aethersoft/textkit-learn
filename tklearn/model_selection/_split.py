import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from tklearn import utils
from tklearn.matrices.scorer import get_score_func

logger = utils.get_logger(__name__)

__all__ = [
    'CrossValidator',
    'train_dev_test_split'
]


class CrossValidator:
    def __init__(self, model, kwargs=None, n_splits=3, scoring=None):
        """Initialize CrossValidator object.

        :param model: The model/estimator to validate.
        :param kwargs: Parameters of model if not initialized.
        :param n_splits: Number of splits.
        :param scoring: Scoring functions.
        """
        self.model = model
        self.kwargs = kwargs
        self.n_splits = n_splits
        self.scoring = scoring if scoring is not None else []

    def cross_validate(self, X, y):
        """Cross-validate input X, y.

        :param X: Input features.
        :param y: Input labels.
        :return: Cross validation results of each split of each scorer.
        """
        return self.cross_val_predict(X, y, return_scores=True)[1]

    def cross_val_predict(self, X, y, return_scores=False):
        """Cross-validate input X, y.

        :param X: Input features.
        :param y: Input labels.
        :param return_scores: Whether to return scoring values of each test-set.
        :return: Cross validation results of each split of each scorer.
        """
        skf = StratifiedKFold(n_splits=self.n_splits)
        predictions = dict()
        split_scores = {scorer: [] for scorer in self.scoring}
        n = 0
        for train_index, test_index in skf.split(np.zeros(len(y)), y):
            if isinstance(X, (pd.Series, pd.DataFrame)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            else:
                y_train, y_test = y[train_index], y[test_index]
            if self.kwargs is None:
                estimator = self.model
            else:
                estimator = self.model(**self.kwargs)
            model = estimator.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.update(dict(zip(test_index, y_pred)))
            for scorer in self.scoring:
                try:
                    split_scores[scorer].append(get_score_func(scorer)(y_test, y_pred))
                except ValueError as ex:
                    logger.warn('Invalid data type for the scorer \'%s\'. %s.' % (scorer, str(ex)))
            n += 1
            logger.info('Training Completed: %i of %i splits.' % (n, self.n_splits))
        predictions = pd.DataFrame.from_dict(predictions, orient='index')
        return (predictions, split_scores) if return_scores else predictions


def train_dev_test_split(X, y, random_state=42):
    """Split arrays or matrices into random train, dev and test subsets.

    :param X: Input features.
    :param y: Input labels.
    :param random_state: Seed used by the random number generator.
    :return: List containing train-test split of inputs.
    """
    # Set random_state to ensure the reproducibility of the samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=random_state)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2,
                                                      random_state=random_state)
    # Reset index
    X_train = X_train.reset_index(drop=True)
    X_dev = X_dev.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_dev = y_dev.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_dev, X_test, y_train, y_dev, y_test
