import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tklearn import utils
from tklearn.matrices.scorer import get_score_func

logger = utils.get_logger(__name__)

__all__ = [
    'CrossValidator',
]


# noinspection PyPep8Naming
class CrossValidator:
    def __init__(self, model, kwargs=None, n_splits=3, scoring=None):
        """ Initialize CrossValidator object.

        Parameters
        ----------
        model
            The model/estimator to validate.

        kwargs
            Parameters of model if not initialized.

        n_splits
            Number of splits.

        scoring
            Scoring functions.
        """
        self.model = model
        self.kwargs = kwargs
        self.n_splits = n_splits
        self.scoring = scoring if scoring is not None else []

    def cross_validate(self, X, y):
        """ Cross-validate input X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be, for example a list, or an array at least 2d.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning.

        Returns
        -------
        cross_validation_scores
            Cross validation results of each split of each scorer.
        """
        return self.cross_val_predict(X, y, return_scores=True)[1]

    def cross_val_predict(self, X, y, return_scores=False):
        """ Cross-validate input X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be, for example a list, or an array at least 2d.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning.

        return_scores : boolean
            Whether to return scoring values of each test-set.

        Returns
        -------
        cross_validation_scores
            Cross validation results of each split of each scorer.

        predictions
            Cross validation predictions.
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
