"""
Metrics to assess performance on regression task
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

# Author: Yasas Senarath <wayasas@gmail.com>
import numpy as np
import scipy.stats
from six import string_types
from sklearn.utils import check_consistent_length, check_array

__ALL__ = [
    'pearson_corr',
    'spearman_corr'
]


def _check_reg_targets(y_true, y_pred):
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    return y_true, y_pred


def pearson_corr(y_true, y_pred, sample_weight=None, multioutput='uniform_average', min_score=0):
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    y_true_min_score = []
    y_pred_min_score = []

    for idx in range(len(y_true)):
        if y_true[idx] >= min_score:
            y_pred_min_score.append(y_pred[idx])
            y_true_min_score.append(y_true[idx])

    # return zero correlation if predictions are constant
    if np.std(y_true_min_score) == 0 or np.std(y_pred_min_score) == 0:
        return 0

    output_scores = scipy.stats.pearsonr(y_pred_min_score, y_true_min_score)[0]

    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_scores
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_scores, weights=multioutput)


def spearman_corr(y_true, y_pred, multioutput='uniform_average', min_score=0):
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    y_true_min_score = []
    y_pred_min_score = []

    for idx in range(len(y_true)):
        if y_true[idx] >= min_score:
            y_true_min_score.append(y_true[idx])
            y_pred_min_score.append(y_pred[idx])

    # return zero correlation if predictions are constant
    if np.std(y_true_min_score) == 0 or np.std(y_pred_min_score) == 0:
        return 0

    output_scores = scipy.stats.spearmanr(y_pred_min_score, y_true_min_score)[0]

    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_scores
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_scores, weights=multioutput)
