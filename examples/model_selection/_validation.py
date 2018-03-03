from tklearn.model_selection import multitask_cross_val_predict
import numpy as np


class SampleFitter:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X, *_):
        return [np.array([X[0][i][0] for i in range(len(X[0]))]), np.array([X[1][i][0] for i in range(len(X[1]))])]


def multi_task_cv_test():
    pred = multitask_cross_val_predict(SampleFitter(),
                                       [np.array([[1, 5], [2, 8], [5, 8], [8, 1]]), np.array([[1], [5], [8]])],
                                       [np.array([1, 5, 8, 9]), np.array([5, 8, 3])])
    assert (pred == [np.array([1, 2, 5, 8]), np.array([1, 5, 8])]).all()


multi_task_cv_test()
