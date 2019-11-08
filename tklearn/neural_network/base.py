import math

import numpy as np
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle, multiclass
from torch import nn
from torch.autograd import Variable

from tklearn import utils
from tklearn.matrices.scorer import get_score_func

logger = utils.get_logger(__name__)

CRITERION = {
    'cross_entropy_loss': nn.CrossEntropyLoss,
    'bce_with_logits_loss': nn.BCEWithLogitsLoss,
    'bce_loss': nn.BCELoss,
    'mse_loss': nn.MSELoss,
}

OPTIMIZER = {
    'adadelta': optim.Adadelta,
}


def get_criterion(criterion, *args, **kwargs):
    if criterion in CRITERION:
        return CRITERION[criterion](*args, **kwargs)
    else:
        return CRITERION['cross_entropy_loss'](*args, **kwargs)


def get_optimizer(optimizer, *args, **kwargs):
    if optimizer in OPTIMIZER:
        return OPTIMIZER[optimizer](*args, **kwargs)
    else:
        return OPTIMIZER['adadelta'](*args, **kwargs)


def load_word_vector(word_vectors, vocab, idx_to_word):
    word_dim = word_vectors.dim
    wv_matrix = []
    for i in range(len(vocab)):
        word = idx_to_word[i]
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, word_dim).astype('float32'))
    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, word_dim).astype('float32'))
    wv_matrix.append(np.zeros(word_dim).astype('float32'))
    wv_matrix = np.array(wv_matrix)
    return wv_matrix


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        Module = kwargs['module']
        # Update Default Parameters from NN Module
        if hasattr(Module, 'DEFAULT_CONFIG'):
            for k, v in Module.DEFAULT_CONFIG.items():
                if k not in kwargs:
                    kwargs[k] = kwargs[k] if k in kwargs else v
        # Extract Information from Corpus
        corpus = kwargs['corpus']
        self.vocab = sorted(list(set([w for sent in corpus for w in sent])))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        kwargs['vocab_size'] = len(self.vocab)
        self.vocab_size = kwargs['vocab_size']
        if 'word_vectors' in kwargs and kwargs['word_vectors'] is not None:
            word_vectors = kwargs['word_vectors']
            kwargs['wv_matrix'] = load_word_vector(word_vectors, self.vocab, self.idx_to_word)
        self.learning_rate = kwargs['learning_rate']
        self.epoch = kwargs['epoch']
        self.batch_size = kwargs['batch_size']
        self.max_sent_len = kwargs['max_sent_len']
        self.norm_limit = kwargs['norm_limit']
        if 'criterion' not in kwargs:
            kwargs['criterion'] = None
        self.criterion = kwargs['criterion']
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = None
        self.optimizer = kwargs['optimizer']
        # Construct Model
        self.model = Module(**kwargs)
        if torch.cuda.is_available():
            logger.info('Found CUDA compatible device. CUDA support is enabled for Neural Network.')
            self.model.cuda()
        else:
            logger.warn('CUDA device not recognized.')
        # Load metrics if available
        self.metrics = kwargs['metrics'] if 'metrics' in kwargs else list()
        self.logs = kwargs['logs'] if 'logs' in kwargs else dict()
        # Parameters assigned while fitting the model
        self.classes_ = None
        self.target_type_ = None

    def fit(self, X, y=None, **kwargs):
        self.target_type_ = multiclass.type_of_target(y)
        self.logs['train_scores'] = []
        self.logs['valid_scores'] = []
        self.classes_ = sorted(list(set(y)))
        callbacks = kwargs['callbacks'] if 'callbacks' in kwargs else []
        # # Get validation data
        valid_x, valid_y = None, None
        if 'validation_data' in kwargs and len(kwargs['validation_data']) > 1:
            if len(kwargs['validation_data']) == 2:
                valid_x, valid_y = kwargs['validation_data']
            else:
                valid_x, valid_y, val_sample_weights = kwargs['validation_data']
        # # Convert to Variable & Move to GPU
        if valid_x is not None and valid_y is not None:
            valid_x = [[self.word_to_idx[w] for w in sent][:self.max_sent_len] + [self.vocab_size + 1] * (
                    self.max_sent_len - len(sent)) for sent in valid_x]
            valid_x = Variable(torch.LongTensor(valid_x))
            if torch.cuda.is_available():
                valid_x = valid_x.cuda()
            valid_y = [self.classes_.index(c) for c in valid_y]
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = get_optimizer(self.optimizer, parameters, self.learning_rate)
        criterion = get_criterion(self.criterion)
        train_x, train_y = X, y
        for e in range(self.epoch):
            self._callback(callbacks, 'epoch_begin')
            progress = None
            train_x, train_y = shuffle(train_x, train_y)
            for i in range(0, len(train_x), self.batch_size):
                self._callback(callbacks, 'batch_begin')
                batch_range = min(self.batch_size, len(train_x) - i)
                # Input as Sequence (`|batch_x| == self.max_sent_len`)
                batch_x = [[self.word_to_idx[w] for w in sent][:self.max_sent_len] + [self.vocab_size + 1] * (
                        self.max_sent_len - len(sent)) for sent in train_x[i:i + batch_range]]
                batch_y = [self.classes_.index(c) for c in train_y[i:i + batch_range]]
                # # Input in torch.Variable
                batch_x = Variable(torch.LongTensor(batch_x))
                batch_y = Variable(torch.LongTensor(batch_y))
                # # Move to GPU
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                # # Back Propagation
                optimizer.zero_grad()
                self.model.train()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm(parameters, max_norm=self.norm_limit)
                optimizer.step()
                # # Calculate training & Validation scores
                # # - Train accuracy is calculated for batch train set
                pred = torch.softmax(pred, dim=1)
                train_score = self._evaluate(batch_y.cpu().data.numpy(), pred.cpu().data.numpy())
                self.logs['train_scores'].append(train_score)
                scores = 'Train: %s' % train_score
                if valid_x is not None and valid_y is not None:
                    valid_pred = self.model(valid_x)
                    valid_pred = torch.softmax(valid_pred, dim=1)
                    valid_score = self._evaluate(valid_y, valid_pred.cpu().data.numpy())
                    self.logs['valid_scores'].append(valid_score)
                    scores = 'Train: %s\t | Valid: %s' % (train_score, valid_score)
                # Show progress
                temp = math.floor((i + self.batch_size) * 20 / len(train_x))
                if temp != progress:
                    progress = temp
                    logger.info("Training Progress: (%i/%i) [%s] %i%% | %s\t" % (
                        e + 1, self.epoch, '=' * progress + '-' * (20 - progress), progress * 5, scores))
                self._callback(callbacks, 'batch_end')
            self._callback(callbacks, 'epoch_end')
        return self

    def predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X, y=None):
        self.model.eval()
        # Input as Sequence [Preprocess Input]
        x = []
        for sent in X:
            idx_seq = [self.word_to_idx[w] if w in self.vocab else self.vocab_size for w in sent]
            padding = [self.vocab_size + 1] * (self.max_sent_len - len(sent))
            x.append(idx_seq[:self.max_sent_len] + padding)
        x = Variable(torch.LongTensor(x))
        if torch.cuda.is_available():  # Move to GPU
            x = x.cuda()
        # Predict Probabilities
        return self.model(x).cpu().data.numpy()

    def _evaluate(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        scores = {scorer: get_score_func(scorer)(y_true, y_pred) for scorer in self.metrics}
        return scores

    def _callback(self, callbacks, status):
        for callback in callbacks:
            if hasattr(callback, 'on_%s' % status):
                callback.on_batch_end(self)
