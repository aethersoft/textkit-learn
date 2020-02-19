import math

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle, multiclass
from tklearn import utils
from tklearn.matrices.scorer import get_score_func

logger = utils.get_logger(__name__)

print('Name: {}'.format(__name__))
logger.info('Logger is Working.')


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
    cuda_available = torch.cuda.is_available()

    def __init__(self, **kwargs):
        if not self.cuda_available:
            logger.warn('CUDA device not recognized.')
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
        self.norm_limit = kwargs['norm_limit']
        if 'criterion' not in kwargs:
            kwargs['criterion'] = None
        self.criterion = kwargs['criterion']
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = None
        self.optimizer = kwargs['optimizer']
        # Construct Model
        self.model = NeuralNetModule(**kwargs)
        if self.cuda_available:
            logger.info('Found CUDA compatible device. CUDA support is enabled for Neural Network.')
            self.model.cuda()
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
        # << Get validation data >>
        valid_x, valid_y = None, None
        if 'validation_data' in kwargs and len(kwargs['validation_data']) > 1:
            if len(kwargs['validation_data']) == 2:
                valid_x, valid_y = kwargs['validation_data']
            else:
                valid_x, valid_y, val_sample_weights = kwargs['validation_data']
        # # Convert to Variable & Move to GPU
        if valid_x is not None:
            valid_x = [[self.word_to_idx[w] for w in sent][:self.max_sent_len] + [self.vocab_size + 1] * (
                    self.max_sent_len - len(sent)) for sent in valid_x]
            valid_x = torch.autograd.Variable(torch.LongTensor(valid_x))
            if self.cuda_available:
                valid_x = valid_x.cuda()
        if valid_y is not None:
            valid_y = [self.classes_.index(c) for c in valid_y]
        # << Get Validation Data / Done >>
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
                x_batch = [
                    [self.word_to_idx[w] for w in sent][:self.max_sent_len] +
                    [self.vocab_size + 1] * (self.max_sent_len - len(sent)) for sent in train_x[i:i + batch_range]
                ]
                y_batch = [self.classes_.index(c) for c in train_y[i:i + batch_range]]
                # # Input in torch.Variable
                x_batch = torch.autograd.Variable(torch.LongTensor(x_batch))
                y_batch = torch.autograd.Variable(torch.LongTensor(y_batch))
                # # Move to GPU
                if self.cuda_available:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                # # Back Propagation
                optimizer.zero_grad()
                self.model.train()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.norm_limit)
                optimizer.step()
                # # Calculate training & Validation scores
                # # - Train accuracy is calculated for batch train set
                y_pred = torch.softmax(y_pred, dim=1)
                train_score = self._evaluate(y_batch.cpu().data.numpy(), y_pred.cpu().data.numpy())
                self.logs['train_scores'].append(train_score)
                scores = 'Train: %s' % train_score
                print(valid_x, valid_y)
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
                    print('Training Progress: (%i/%i) [%s] %i%% | %s\t' % (
                        e + 1, self.epoch, '=' * progress + '-' * (20 - progress), progress * 5, scores))
                self._callback(callbacks, 'batch_end')
            self._callback(callbacks, 'epoch_end')
        return self

    def predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X, y=None):
        self.model.eval()
        # Input as Sequence [Preprocess Input]
        x = [[self.word_to_idx[w] if w in self.vocab else self.vocab_size for w in sent][:self.max_sent_len] +
             [self.vocab_size + 1] * (self.max_sent_len - len(sent)) for sent in X]
        x = torch.autograd.Variable(torch.LongTensor(x))
        if self.cuda_available:  # Move to GPU
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
