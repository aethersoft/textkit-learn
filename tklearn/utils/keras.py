from keras.callbacks import Callback

from tklearn.metrics.scorer import check_scoring


class ValidationLogger(Callback):
    def __init__(self, validation_data, scorer='accuracy'):
        super(ValidationLogger, self).__init__()
        self._validation_data = validation_data
        self._validation_scorer = scorer

    def on_epoch_end(self, epoch, logs=None):
        X, y_true = self._validation_data
        y_pred = self.model.predict(X)
        score = check_scoring(self._validation_scorer)(y_true, y_pred)
        print('\nTesting {}: {}\n'.format(self._validation_scorer, score))
