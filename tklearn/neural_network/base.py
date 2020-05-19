import math

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle, multiclass
from tklearn import utils
from tklearn.matrices.scorer import get_score_func
from tklearn.neural_network.utils import move_to_device, move_from_device

logger = utils.get_logger(__name__)


def get_criterion(criterion, *args, **kwargs):
    """ Returns loss function based on the provided string criterion name.

    :param criterion: name of the criterion.
    :param args: args passed to criterion constructor
    :param kwargs: kwargs passed to criterion constructor
    :return: Loss instance
    """
    if criterion == 'bce_with_logits_loss':
        return torch.nn.BCEWithLogitsLoss(*args, **kwargs)
    elif criterion == 'bce_loss':
        return torch.nn.BCELoss(*args, **kwargs)
    elif criterion == 'mse_loss':
        return torch.nn.MSELoss(*args, **kwargs)
    else:  # cross_entropy_loss (DEFAULT)
        return torch.nn.CrossEntropyLoss(*args, **kwargs)


def get_optimizer(optimizer, *args, **kwargs):
    """ Gets optimizer based on the text optimiser provided.
    As of now optimizer argument is ignored and the function always returns `Adadelta`.

    :param optimizer: ignore
    :param args: args passed to optimizer constructor
    :param kwargs: kwargs passed to optimizer constructor
    :return: Adadelta optimizer instance
    """
    return torch.optim.Adadelta(*args, **kwargs)


def load_word_vector(word_embedding, vocab, idx_to_word):
    word_dim = word_embedding.dim
    embedding_matrix = []
    for i in range(len(vocab)):
        word = idx_to_word[i]
        if word in word_embedding.vocab:
            embedding_matrix.append(word_embedding.word_vec(word))
        else:
            embedding_matrix.append(np.random.uniform(-0.01, 0.01, word_dim).astype('float32'))
    # one for UNK and one for zero padding
    embedding_matrix.append(np.random.uniform(-0.01, 0.01, word_dim).astype('float32'))
    embedding_matrix.append(np.zeros(word_dim).astype('float32'))
    embedding_matrix = np.array(embedding_matrix)
    return embedding_matrix


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    device = ('cpu', 'cuda')[torch.cuda.is_available()]

    def __init__(self, **kwargs):
        logger.debug('Using device %s for the model.' % self.device)
        # noinspection PyPep8Naming
        NeuralNetModule = kwargs['module']
        # Update Default Parameters from NN Module
        if hasattr(NeuralNetModule, 'DEFAULT_CONFIG'):
            for k, v in NeuralNetModule.DEFAULT_CONFIG.items():
                if k not in kwargs:
                    kwargs[k] = kwargs[k] if k in kwargs else v
        # Extract Information from Corpus
        corpus = kwargs['corpus']
        self.vocab = sorted(list(set([w for sent in corpus for w in sent])))
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        kwargs['vocab_size'] = self.vocab_size
        if 'word_embedding' in kwargs and kwargs['word_embedding'] is not None:
            word_embedding = kwargs['word_embedding']
            kwargs['embedding_matrix'] = load_word_vector(word_embedding, self.vocab, self.idx_to_word)
        self.learning_rate = kwargs['learning_rate']
        self.epoch = kwargs['epoch']
        self.batch_size = kwargs['batch_size']
        self.max_sent_len = kwargs['max_sent_len']
        self.clip_value = kwargs['clip_value']
        if 'criterion' not in kwargs:
            kwargs['criterion'] = None
        self.criterion = kwargs['criterion']
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = None
        self.optimizer = kwargs['optimizer']
        kwargs['device'] = self.device
        # Construct Model
        self.model = NeuralNetModule(**kwargs)
        move_to_device(self.device, self.model)
        # Load metrics if available
        self.metrics = kwargs['metrics'] if 'metrics' in kwargs else list()
        self.logs = kwargs['logs'] if 'logs' in kwargs else dict()
        # Parameters assigned while fitting the model
        self.label_enc_ = None
        self.target_type_ = None

    def fit(self, X, y=None, validation_data=None, callbacks=None, **kwargs):
        """ Fit Neural Network model.
        Notes: Works for binary target only.

        :param X: Training data
        :param y: Target values
        :param validation_data: Validation Data. Tuple containing validation features and target pairs
        :param callbacks: Callbacks
        :param kwargs: Other args (ignored)
        :return: returns an instance of self.
        """
        self.target_type_ = multiclass.type_of_target(y)
        self.logs = []
        self.label_enc_ = preprocessing.LabelEncoder()
        self.label_enc_.fit(y)
        if callbacks is None:
            callbacks = []
        # << Get validation data >>
        valid_x, valid_y = None, None
        if validation_data is not None and len(validation_data) > 1:
            if len(validation_data) == 2:
                valid_x, valid_y = validation_data
            else:
                valid_x, valid_y, val_sample_weights = validation_data
        # # Convert to Variable & Move to GPU
        valid_x_d = None
        if valid_x is not None:
            valid_x = [[self.word_to_idx[w] for w in sent][:self.max_sent_len] + [self.vocab_size + 1] * (
                    self.max_sent_len - len(sent)) for sent in valid_x]
            valid_x = torch.autograd.Variable(torch.LongTensor(valid_x))
            valid_x_d = move_to_device(self.device, valid_x)
        if valid_y is not None:
            valid_y = self.label_enc_.transform(valid_y)
        # << Get Validation Data / Done >>
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = get_optimizer(self.optimizer, parameters, self.learning_rate)
        criterion = get_criterion(self.criterion)
        train_x, train_y = X, y
        for e in range(self.epoch):
            self._callback(callbacks, 'epoch_begin')
            progress = None
            train_x, train_y = shuffle(train_x, train_y)
            for batch, i in enumerate(range(0, len(train_x), self.batch_size)):
                self._callback(callbacks, 'batch_begin')
                batch_range = min(self.batch_size, len(train_x) - i)
                # Input as Sequence (`|batch_x| == self.max_sent_len`)
                x_batch = [
                    [self.word_to_idx[w] for w in sent][:self.max_sent_len] +
                    [self.vocab_size + 1] * (self.max_sent_len - len(sent)) for sent in train_x[i:i + batch_range]
                ]
                y_batch = self.label_enc_.transform(train_y[i:i + batch_range])
                # # Input in torch.Variable
                x_batch = torch.autograd.Variable(torch.LongTensor(x_batch))
                x_batch_d = move_to_device(self.device, x_batch)
                y_batch = torch.autograd.Variable(torch.LongTensor(y_batch))
                y_batch_d = move_to_device(self.device, y_batch)
                # # Back Propagation
                optimizer.zero_grad()
                self.model.train()
                y_pred_d = self.model(x_batch_d)
                loss = criterion(y_pred_d, y_batch_d)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                optimizer.step()
                # # Calculate training & Validation scores
                # # - Train accuracy is calculated for batch train set
                y_pred_d = torch.softmax(y_pred_d, dim=1)
                train_score = self._evaluate(y_batch, move_from_device(self.device, y_pred_d).data.numpy())
                self.logs.append({'key': 'train_score', 'value': train_score, 'epoch': e, 'batch': batch})
                scores = 'train_score: %s' % train_score
                if valid_x_d is not None and valid_y is not None:
                    valid_pred_d = torch.softmax(self.model(valid_x_d), dim=1)
                    valid_score = self._evaluate(valid_y, move_from_device(self.device, valid_pred_d).data.numpy())
                    self.logs.append({'key': 'validate_score', 'value': valid_score, 'epoch': e, 'batch': batch})
                    scores += '| validation_score: %s' % valid_score
                # Show progress
                temp = math.floor((i + self.batch_size) * 20 / len(train_x))
                if temp != progress:
                    progress = temp
                    print('Training Progress: (%i/%i) [%s] %i%% | %s\t' % (
                        e + 1, self.epoch, '=' * progress + '-' * (20 - progress), progress * 5, scores))
                self._callback(callbacks, 'batch_end')
            self._callback(callbacks, 'epoch_end')
        return self

    def predict(self, X, y=None):
        return self.label_enc_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def predict_proba(self, X, y=None):
        self.model.eval()
        # Input as Sequence [Preprocess Input]
        x = [[self.word_to_idx[w] if w in self.vocab else self.vocab_size for w in sent][:self.max_sent_len] +
             [self.vocab_size + 1] * (self.max_sent_len - len(sent)) for sent in X]
        x = torch.autograd.Variable(torch.LongTensor(x))
        x = move_to_device(self.device, x)
        # Predict Probabilities
        return move_from_device(self.device, self.model(x)).data.numpy()

    def _evaluate(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        scores = {scorer: get_score_func(scorer)(y_true, y_pred) for scorer in self.metrics}
        return scores

    def _callback(self, callbacks, status):
        for callback in callbacks:
            if hasattr(callback, 'on_%s' % status):
                callback.on_batch_end(self)
