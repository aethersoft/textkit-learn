from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score


class ValidationLogger(Callback):
    def __init__(self, validation_data, scorer='accuracy'):
        super(ValidationLogger, self).__init__()
        self.validation_data = validation_data
        self.scorer = scorer

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        x, y = self.validation_data
        if self.scorer == 'micro_f1':
            scorer = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
        else:
            scorer = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
        y_pred = self.model.predict(x)
        score = scorer(y, y_pred)
        print('\nTesting {}: {}\n'.format(self.scorer, score))
