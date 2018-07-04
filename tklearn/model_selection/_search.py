import logging
import time

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, rand
from hyperopt.pyll import scope

logger = logging.getLogger(__name__)

__all__ = [
    'HyperoptOptimizer'
]


def _build_space(specs, name=''):
    if specs is None:
        raise TypeError('Invalid search space specification.')
    val_type = specs['type'] if 'type' in specs else None
    val_val = specs['value'] if 'value' in specs else None
    if val_type == 'rlist':
        length = specs['length']
        val = []
        for j in range(length[0], length[1] + 1, 1):
            lst = []
            for i in range(j):
                lst.append(_build_space(val_val, '{}_{}_{}'.format(name, j, i)))
            val.append(lst)
        return hp.choice(name, val)
    elif val_type == 'dict':
        val = {}
        for k, v in val_val.items():
            val[k] = _build_space(v, name=k)
        return val
    elif val_type == 'list':
        val = []
        for i, item in enumerate(val_val):
            val.append(_build_space(item, name='{}_{}'.format(name, i)))
        return val
    elif val_type == 'choice':
        val = []
        for i, v in enumerate(val_val):
            val.append(_build_space(v, name='{}_{}'.format(name, i)))
        return hp.choice(name, val)
    elif val_type == 'uniform':
        if len(val_val) == 2:
            val = hp.uniform(name, val_val[0], val_val[1])
        elif len(val_val) == 3:
            val = hp.quniform(name, val_val[0], val_val[1], val_val[2])
        if isinstance(val_val[0], int) and isinstance(val_val[1], int):
            val = scope.int(val)
        return val
    elif val_type == 'var':
        return val_val
    else:
        logger.warning('Ignoring invalid value type and continuing.')
    return []


class HyperoptOptimizer:
    def __init__(self, estimator, param_dist, scorer, max_evals=25, search_algorithm='tpe', callbacks=None):
        """
        Initialize Hyperopt Optimizer object.
        :param estimator: Estimator object; A object of that type is instantiated for each evaluation point.
        :param param_dist: Description of the parameters
        :param max_evals: Allow up to this many function evaluations before returning.
        """
        self.estimator = estimator
        self.param_dist = param_dist
        self.scorer = scorer
        self.max_evals = max_evals
        self.search_algorithm = search_algorithm
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    def optimize(self, X_train, X_test, y_train, y_test):
        trials = Trials()
        space = _build_space({
            'type': 'dict',
            'value': self.param_dist
        })

        def objective(kwargs):
            logger.info('Training model with parameters: {}'.format(kwargs))
            y_pred = self.estimator(**kwargs).fit(X_train, y_train).predict(X_test)
            loss = -self.scorer(y_test, y_pred)
            for func in self.callbacks:
                func({'params': kwargs, 'loss': loss, 'y_test': y_test, 'y_pred': y_pred})
            return {
                'loss': loss,
                'status': STATUS_OK,
                'eval_time': time.time(),
            }

        if self.search_algorithm == 'rand':
            algo = rand.suggest
        else:
            algo = tpe.suggest
        fmin(objective, space=space, algo=algo, max_evals=self.max_evals, trials=trials)
        return trials
