import json
import os
from abc import ABC, abstractmethod

import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


# ======================================================================================================================
# Classifier -----------------------------------------------------------------------------------------------------------

class KerasClassifier(ABC, BaseEstimator, ClassifierMixin):
    """
    An more abstract version of KerasClassifier for scikit-learn with custom preprocess pipeline.
    """

    def __init__(self, batch_size=16, epochs=8, output='classify'):
        """
        Initialize the Classifier.

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
        if hasattr(self, '_transfer'):
            X, _ = self.preprocess(X)
            return self._features(self._tlayer if hasattr(self, '_tlayer') else None).predict(X)
        y = self.predict_proba(X)
        return y if self.output == 'multilabel' else np.argmax(y, axis=1, out=None)  # multiclass/binary

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
            json.dump({'batch_size': self.batch_size, 'epochs': self.epochs, 'output': self.output}, f)

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

    def _features(self, layer_name=None):
        if layer_name is None:
            layer_name = self.model_.layers[-2].name
        intermediate_layer_model = Model(inputs=self.model_.input,
                                         outputs=self.model_.get_layer(layer_name).output)
        return intermediate_layer_model


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
            return self._features(self._tlayer if hasattr(self, '_tlayer') else None).predict(X)
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

    def _features(self, layer_name=None):
        if layer_name is None:
            layer_name = self.model_.layers[-2].name
        intermediate_layer_model = Model(inputs=self.model_.input,
                                         outputs=self.model_.get_layer(layer_name).output)
        return intermediate_layer_model
