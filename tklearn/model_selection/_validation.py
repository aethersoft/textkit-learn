import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv
from sklearn.utils import indexable

from tklearn.utils import concatenate_array, invert_array, apply

__all__ = ['multitask_cross_val_predict']


def _multitask_fit_predict(estimator, X, y, train, test):
    x_train, y_train, x_test, y_test_idx = [], [], [], []
    _idx = 0
    for _x, _y, _train, _test in zip(X, y, train, test):
        #  convert to numpy arrays if instance is list
        if isinstance(_y, list):
            _y = np.array(_y)
        if isinstance(_x, list):
            _x = np.array(_x)
        x_train += [_x[_train]]
        x_test += [_x[_test]]
        if isinstance(_y, list):
            _y = np.array(_y)
        if _y is not None:
            y_train += [_y[_train]]
            y_test_idx += [_test]
        else:
            pass
        _idx += 1
    y_pred = estimator.fit(x_train, y_train).predict(x_test)
    return y_pred, y_test_idx


def _multitask_indexable(X, y, groups=None):
    t1, t2, t3 = [], [], []
    for _x, _y in zip(X, y):
        __x, __y, __g = indexable(_x, _y, groups)
        t1 += [__x]
        t2 += [__y]
        t3 += [__g]
    return t1, t2, t3


def multitask_cross_val_predict(estimator, X, y, groups=None, cv=2):
    """
    Divides a list of inputs (Xs) and parallel labels (ys) to folds as provided by cv parameter and predicts output
     for each task. The associated pipeline/ estimator/ transformer should be able to handle input with multiple tasks.

    :param estimator: an estimator/ pipeline with a last stage implementing predict function
    :param X: input data for each task
    :param y: parallel label sequence matching tasks in input
    :param groups: [ignored]
    :param cv: an integer indicating the number of folds or the cross validation iterators as provided in scikit-lean library
    :return: predictions for each task as a list
    """
    assert len(X) == len(y), 'Cross validation requires a parallel data and label dataset. ' \
                             'Please fill \'None\' data-points explicitly.'
    X, y, groups = _multitask_indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    cv_splits = [([], []) for _ in range(cv.n_splits)]
    for _X, _y in zip(X, y):
        _idx = 0
        for train, test in cv.split(_X, _y):
            cv_splits[_idx][0].append(train)
            cv_splits[_idx][1].append(test)
            _idx += 1
    predictions_splits = [(_multitask_fit_predict(estimator, X, y, train, test),) for train, test in cv_splits]
    predictions = None
    prediction_idx = None
    # Concatenate the predictions
    for predictions_split in predictions_splits:
        for block_y, block_y_index in predictions_split:
            _expected_len = len(block_y_index)
            _output_len = len(block_y)
            if _expected_len != _output_len:
                raise ValueError(
                    'Output of the estimator is incompatible with size of the input: '
                    'expected len={}, found len={}'.format(_expected_len, _output_len))
            if block_y is not None:
                if predictions is None:
                    predictions = [[] for _ in range(_expected_len)]
                    prediction_idx = [[] for _ in range(_expected_len)]
                for i, (block_y_task_i, block_y_index_task_i) in enumerate(zip(block_y, block_y_index)):
                    predictions[i] += [block_y_task_i]
                    prediction_idx[i] += [block_y_index_task_i]
    prediction_idx = apply(invert_array, apply(concatenate_array, prediction_idx))
    predictions = apply(concatenate_array, predictions)
    out = [p[idx] for p, idx in zip(predictions, prediction_idx)]
    return out
