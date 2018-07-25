import json
import os
from abc import ABC, abstractmethod

import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import model_from_json
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# ======================================================================================================================
# Classifier -----------------------------------------------------------------------------------------------------------
from tklearn.utils.keras import ValidationLogger


class KerasClassifier(ABC, BaseEstimator, ClassifierMixin):
    """
    An more abstract version of KerasClassifier for scikit-learn with custom preprocess pipeline.
    """

    def __init__(self, batch_size=16, epochs=8):
        """
        Initialize the Classifier.

        :param batch_size: Number of samples per gradient update. If unspecified, it will default to 16
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self._return_probs = False

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data

        :param X: Training vector with  shape = [n_samples, n_features],
         where n_samples in the number of samples and  n_features is the number of features.
        :param y: Target vector relative to X of shape shape = [n_samples]
        :return: self
        """
        X, y = self.preprocess(X, y)
        self.model_ = self.build_model(X, y)
        callbacks = []
        if hasattr(self, '_early_stopping'):
            callbacks.append(EarlyStopping(**self._early_stopping))
        if hasattr(self, '_validation_data') and hasattr(self, '_validation_scorer'):
            callbacks.append(ValidationLogger(self._validation_data, self._validation_scorer))
        if not hasattr(self, '_log_dir'):
            self.log_dir('./logs')
        tb = TensorBoard(log_dir=self._log_dir, histogram_freq=0, batch_size=32, write_graph=True,
                         write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None)
        callbacks.append(tb)
        if len(callbacks) == 0:
            callbacks = None
        validation_split = 0.0
        if hasattr(self, '_validation_split'):
            validation_split = self._validation_split
        self.model_.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks,
                        validation_split=validation_split)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X / Predict target values of X given a model

        :param X: Testing vector
        :return: Predicted values
        """
        if hasattr(self, '_transfer'):
            X, _ = self.preprocess(X)
            if hasattr(self, '_tlayer'):
                if isinstance(self._tlayer, int):
                    return self._features(layer_index=self._tlayer).predict(X)
                return self._features(layer_name=self._tlayer).predict(X)
            else:
                return self._features().predict(X)
        y = self.predict_proba(X)
        if self._return_probs:
            return y
        else:
            return np.argmax(y, axis=1, out=None)

    # predicts probability output
    def predict_proba(self, X):
        X, _ = self.preprocess(X)
        return self.model_.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy on the given test data and labels.

        :param X: Test samples
        :param y: True labels for X
        :param sample_weight: Sample weights
        :return:
        """
        _, acc = self.evaluate(X, y, sample_weight)
        return acc

    def evaluate(self, x_train, y, sample_weight=None):
        """
        Returns the (loss, accuracy) tuple on the given test data and labels.

        :param x_train: Test samples with  shape = [n_samples, n_features],
         where n_samples in the number of samples and  n_features is the number of features.
        :param y: True labels for X of shape shape = [n_samples]
        :param sample_weight: Sample weights.
        :return:
        """
        x_train, y = self.preprocess(x_train, y)
        return self.model_.evaluate(x_train, y, batch_size=self.batch_size)

    @abstractmethod
    def build_model(self, X, y):
        """
        Compiles the CNN model

        :return: compiled model
        """
        pass

    @abstractmethod
    def preprocess(self, X, y=None):
        """
        Preprocess the data and return the preprocessed input for the model

        :return: model input
        """
        return X, y

    def transfer(self, status, layer=None):
        self._transfer = status
        self._tlayer = layer

    def early_stopping(self, early_stopping, monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto',
                       **kwargs):
        if early_stopping:
            self._early_stopping = {
                'monitor': monitor,
                'min_delta': min_delta,
                'patience': patience,
                'verbose': verbose,
                'mode': mode,
            }
        else:
            del self.__dict__["_early_stopping"]

    def validation_split(self, validation_split, **kwargs):
        self._validation_split = validation_split

    def validation_logger(self, validation_data, validation_scorer, **kwargs):
        self._validation_data = validation_data
        self._validation_scorer = validation_scorer

    def _features(self, layer_name=None, layer_index=-2):
        if layer_name is None:
            layer_name = self.model_.layers[layer_index].name
        intermediate_layer_model = Model(inputs=self.model_.input, outputs=self.model_.get_layer(layer_name).output)
        return intermediate_layer_model

    def log_dir(self, value='./logs'):
        self._log_dir = value

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    def save(self, path):
        """
        Saves the model weights in a file provided by path
        :type filepath: path to the file
        :return: status
        """
        model_path = os.path.join(path, 'model.json')
        weights_path = os.path.join(path, 'weights.h5')
        args_path = os.path.join(path, 'args.json')
        # Save the weights
        self.model_.save_weights(weights_path)
        # Save the model architecture
        with open(model_path, 'w') as f:
            f.write(self.model_.to_json())
        with open(args_path, 'w') as f:
            json.dump({'batch_size': self.batch_size, 'epochs': self.epochs, '_return_probs': self._return_probs}, f)

    def load(self, path):
        """
        Loads model from the filepath
        :type filepath: path to the file
        :return: Saved model
        """
        model_path = os.path.join(path, 'model.json')
        weights_path = os.path.join(path, 'weights.h5')
        args_path = os.path.join(path, 'args.json')
        if not os.path.isfile(model_path):
            raise IOError('Ca\'t find models in \'{}\''.format(model_path))
        if not os.path.isfile(weights_path):
            raise IOError('Ca\'t find weights in \'{}\''.format(weights_path))
        # Model reconstruction from JSON file
        with open(model_path, 'r') as f:
            self.model_ = model_from_json(f.read())
        # Load weights into the new model
        self.model_.load_weights(weights_path)
        with open(args_path, 'r') as f:
            kwargs = json.load(f)
            self.batch_size = kwargs['batch_size']
            self.epochs = kwargs['epochs']
            self._return_probs = kwargs['_return_probs']


# ======================================================================================================================
# Regressor ------------------------------------------------------------------------------------------------------------

class KerasRegressor(ABC, BaseEstimator, RegressorMixin):
    """
    An more abstract version of KerasRegressor for scikit-learn with custom preprocess pipeline.
    """

    def __init__(self, batch_size=16, epochs=8, output='predict'):
        """
        Initialize the Regressor.

        :param batch_size: Number of samples per gradient update. If unspecified, it will default to 16
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.output = output

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data

        :param X: Training vector with  shape = [n_samples, n_features],
         where n_samples in the number of samples and  n_features is the number of features.
        :param y: Target vector relative to X of shape shape = [n_samples]
        :return: self
        """
        X, y = self.preprocess(X, y)
        self.model_ = self.build_model(X, y)
        callbacks = []
        if hasattr(self, '_early_stopping'):
            callbacks.append(EarlyStopping(**self._early_stopping))
        validation_split = 0.0
        if hasattr(self, '_validation_split'):
            validation_split = self._validation_split
        if hasattr(self, '_validation_data') and hasattr(self, '_validation_scorer'):
            callbacks.append(ValidationLogger(self._validation_data, self._validation_scorer))
        if len(callbacks) == 0:
            callbacks = None
        self.model_.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks,
                        validation_split=validation_split)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X / Predict target values of X given a model

        :param X: Testing vector
        :return: Predicted values
        """
        X, _ = self.preprocess(X)
        if hasattr(self, '_transfer'):
            if hasattr(self, '_tlayer'):
                if isinstance(self._tlayer, int):
                    return self._features(layer_index=self._tlayer).predict(X)
                return self._features(layer_name=self._tlayer).predict(X)
            else:
                return self._features().predict(X)
        return self.model_.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy on the given test data and labels.

        :param X: Test samples
        :param y: True labels for X
        :param sample_weight: Sample weights
        :return:
        """
        _, acc = self.evaluate(X, y, sample_weight)
        return acc

    def evaluate(self, x_train, y, sample_weight=None):
        """
        Returns the (loss, accuracy) tuple on the given test data and labels.

        :param x_train: Test samples with  shape = [n_samples, n_features],
         where n_samples in the number of samples and  n_features is the number of features.
        :param y: True labels for X of shape shape = [n_samples]
        :param sample_weight: Sample weights.
        :return:
        """
        x_train, y = self.preprocess(x_train, y)
        return self.model_.evaluate(x_train, y, batch_size=self.batch_size)

    @abstractmethod
    def build_model(self, X, y):
        """
        Compiles the CNN model

        :return: compiled model
        """
        pass

    def save(self, path):
        """
        Saves the model weights in a file provided by path
        :type path: path to the file
        :return: status
        """
        model_path = os.path.join(path, 'model.json')
        args_path = os.path.join(path, 'args.json')
        weights_path = os.path.join(path, 'weights.h5')
        # Save the weights
        self.model_.save_weights(weights_path)
        # Save the model architecture
        with open(model_path, 'w') as f:
            f.write(self.model_.to_json())
        with open(args_path, 'w') as f:
            json.dump({'batch_size': self.batch_size, 'epochs': self.epochs, 'output': self.output}, f)

    def load(self, path):
        """
        Loads model from the filepath
        :type path: path to the file
        :return: Saved model
        """
        model_path = os.path.join(path, 'model.json')
        weights_path = os.path.join(path, 'weights.h5')
        args_path = os.path.join(path, 'args.json')
        if not os.path.isfile(model_path):
            raise IOError('Ca\'t find models in \'{}\''.format(model_path))
        if not os.path.isfile(weights_path):
            raise IOError('Ca\'t find weights in \'{}\''.format(weights_path))
        # Model reconstruction from JSON file
        with open(model_path, 'r') as f:
            self.model_ = model_from_json(f.read())
        # Load weights into the new model
        self.model_.load_weights(weights_path)
        with open(args_path, 'r') as f:
            kwargs = json.load(f)
            self.batch_size = kwargs['batch_size']
            self.epochs = kwargs['epochs']
            self.output = kwargs['output']

    @abstractmethod
    def preprocess(self, X, y=None):
        """
        Preprocess the data and return the preprocessed input for the model

        :return: model input
        """
        return X, y

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    def transfer(self, status, layer=None):
        self._transfer = status
        self._tlayer = layer

    def early_stopping(self, early_stopping, monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto',
                       **kwargs):
        if early_stopping:
            self._early_stopping = {
                'monitor': monitor,
                'min_delta': min_delta,
                'patience': patience,
                'verbose': verbose,
                'mode': mode,
            }
        else:
            del self.__dict__["_early_stopping"]

    def validation_split(self, validation_split, **kwargs):
        self._validation_split = validation_split

    def validation_logger(self, validation_data, validation_scorer, **kwargs):
        self._validation_data = validation_data
        self._validation_scorer = validation_scorer

    def _features(self, layer_name=None, layer_index=-2):
        if layer_name is None:
            layer_name = self.model_.layers[layer_index].name
        intermediate_layer_model = Model(inputs=self.model_.input, outputs=self.model_.get_layer(layer_name).output)
        return intermediate_layer_model
