from sklearn.utils.multiclass import type_of_target


def optimize_threshold(y_true, y_pred, scorer, **kwargs):
    """
    Optimize threshold for ``y_pred`` if ``y_pred`` is a continuous multioutput while ``y_true`` is multilabel

    :param y_true: array-like
    :param y_pred: array-like
    :param scorer: a scorer to optimize predictions on
    :param kwargs: extra arguments for the provided scorer
    :return: categorical y_pred (if not already)
    """
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        if type_pred.startswith('continuous') and (
                type_true.startswith('multilabel') or type_true.startswith('binary')):
            score = 0
            threshold = 0.5
            for i in range(50, 0, -1):
                t = i / 100
                s = scorer(y_true, y_pred >= t, **kwargs)
                if s > score:
                    threshold = t
                    score = s
            y_pred = y_pred >= threshold
    return y_pred
