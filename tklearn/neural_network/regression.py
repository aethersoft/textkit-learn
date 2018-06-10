import keras
import numpy as np
from keras import Model, Input, Sequential
from keras.engine import InputLayer
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    Dropout, LSTM, Dense, MaxPooling1D, AveragePooling1D
from keras.utils import to_categorical
from scipy.sparse import isspmatrix

from .base import KerasRegressor


class FNNRegressor(KerasRegressor):
    def __init__(self, hidden_dims, batch_size=16, epochs=8):
        super(FNNRegressor, self).__init__(batch_size, epochs)
        self.hidden_dims = hidden_dims

    def preprocess(self, X, y=None):
        if isspmatrix(X):
            X = np.array(X.todense())
        assert len(X) >= 1, 'Sample size should be grater than or equal to 1 found {}'.format(len(X))
        if not hasattr(self, 'num_features_'):
            self.num_features_ = len(X[0])
        return X, y

    def build_model(self, X, y):
        """
         Compiles the FNN model

         :return: compiled classifier
         """
        model = Sequential()

        model.add(InputLayer(input_shape=(self.num_features_,), sparse=False, dtype='float32'))

        for dim in self.hidden_dims:
            # Hidden layer:
            if isinstance(dim, tuple):
                assert len(dim) == 2, 'Invalid parameter input valued \'{}\' for hidden dimensions. ' \
                                      'This parameter can be an integer or a tuple of dimension 2.'.format(dim)
                model.add(Dense(dim[0], activation='relu'))
                model.add(Dropout(dim[1]))
            else:
                model.add(Dense(dim, activation='relu'))

        # Output layer with sigmoid activation:
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model before use
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
        return model


class CNNRegressor(KerasRegressor):
    def __init__(self, filters=250, kernel_size=3, pooling=None, dropout=None,
                 hidden_dims=None, trainable=False, batch_size=32, epochs=15):
        """
        Initializes the classifier

        :param filters: Number output of filters in the convolution.
        :param kernel_size: Length of the 1D convolution window.
        :param pooling: Type of pooling; pooling is not done if None.
        :param dropout:
        :param hidden_dims: List of dimensionality of the output space of the dense layers. None will be default to [250, 50].
        :param trainable: Whether cnn is trainable or not
        :param batch_size: Number of samples per gradient update. If unspecified, it will default to 32.
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        """
        super(CNNRegressor, self).__init__(batch_size, epochs)
        self.filters = filters
        self.pooling = pooling
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.trainable = trainable
        self.initialize()

    def initialize(self):
        if self.hidden_dims is None:
            self.hidden_dims = []
        else:
            self.hidden_dims = list(self.hidden_dims)
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size]

    def preprocess(self, X, y=None):
        if not hasattr(self, 'embedding_matrix_'):
            self.embedding_matrix_ = X['embedding_matrix']
        if not hasattr(self, 'vocab_size_'):
            self.vocab_size_ = self.embedding_matrix_.shape[0]
        if not hasattr(self, 'sequence_length_'):
            self.sequence_length_ = X['tokens'].shape[1]
        X = X['tokens']
        return X, y

    def build_model(self, X, y):
        """
        Compiles the CNN model

        :return: compiled classifier
        """
        ipt = Input(shape=(self.sequence_length_,), sparse=False, dtype='int32')
        opt = ipt

        # Embedding layer (Extracts embedding from embedding matrix according to input index sequence)
        e0 = Embedding(self.vocab_size_, self.embedding_matrix_.shape[1], weights=[self.embedding_matrix_],
                       input_length=self.sequence_length_, trainable=self.trainable)
        opt = e0(opt)

        p1 = []
        for ks in self.kernel_size:
            # Convolution1D Layer
            c0 = Conv1D(self.filters, ks, padding='valid', activation='relu', strides=1)(opt)
            # Pooling layer:
            p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(self.pooling, c0)
            p1.append(p0)
        if len(p1) > 1:
            opt = keras.layers.concatenate(p1, axis=1)
        else:
            opt = p1[0]

        # Dropout layers
        if self.dropout is not None:
            d0 = Dropout(self.dropout)
            opt = d0(opt)

        # Hidden layers
        for dim in self.hidden_dims:
            l0 = Dense(dim, activation='relu')
            opt = l0(opt)

        # Output layer with sigmoid activation:
        opt = Dense(1, activation='sigmoid')(opt)

        model = Model(inputs=ipt, outputs=opt)

        # Compile
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
        return model


class LSTMRegressor(KerasRegressor):
    def __init__(self, trainable=False, lstm_units=150, hidden_dims=None, batch_size=16, epochs=8):
        super(LSTMRegressor, self).__init__(batch_size, epochs)
        self.trainable = trainable
        self.hidden_dims = hidden_dims
        self.lstm_units = lstm_units
        self.initialize()

    def initialize(self):
        if self.hidden_dims is None:
            self.hidden_dims = []
        else:
            self.hidden_dims = list(self.hidden_dims)

    def preprocess(self, X, y=None):
        tokens = X['tokens']
        if not hasattr(self, 'embedding_matrix_'):
            self.embedding_matrix_ = X['embedding_matrix']
        if not hasattr(self, 'sequence_length_'):
            self.sequence_length_ = tokens.shape[1]
        if not hasattr(self, 'vocab_size_'):
            self.vocab_size_ = self.embedding_matrix_.shape[0]
        return tokens, y

    def build_model(self, X, y):
        """
         Compiles the lstm model

         :return: compiled classifier
         """
        model = Sequential()

        model.add(InputLayer(input_shape=(self.sequence_length_,), sparse=False, dtype='int32'))
        model.add(Embedding(self.vocab_size_,
                            self.embedding_matrix_.shape[1],
                            weights=[self.embedding_matrix_],
                            input_length=self.sequence_length_,
                            trainable=self.trainable))
        # Add LSTM layer
        model.add(LSTM(units=self.lstm_units))
        # MLPs
        for dim in self.hidden_dims:
            # Hidden layer:
            if isinstance(dim, tuple):
                assert len(dim) == 2, 'Invalid parameter input valued \'{}\' for hidden dimensions. ' \
                                      'This parameter can be an integer or a tuple of dimension 2.'.format(dim)
                model.add(Dense(dim[0], activation='relu'))
                model.add(Dropout(dim[1]))
            else:
                model.add(Dense(dim, activation='relu'))

        # Output layer with sigmoid activation:
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model before use
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
        return model


class CNNLSTMRegressor(KerasRegressor):
    def __init__(self, filters=250, kernel_size=3, pooling=None, dropout=None, lstm_units=300,
                 pool_size=1, hidden_dims=None, trainable=False, batch_size=32, epochs=15):
        """
        Initializes the classifier

        :param filters: Number output of filters in the convolution.
        :param kernel_size: Length of the 1D convolution window.
        :param pooling: Type of pooling; pooling is not done if None.
        :param dropout: dropout after cnn layer.
        :param lstm_units: number of lstm units (output size)
        :param hidden_dims: List of dimensionality of the output space of the dense layers. None will be default to [250, 50].
        :param trainable: Whether cnn is trainable or not
        :param batch_size: Number of samples per gradient update. If unspecified, it will default to 32.
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        """
        super(CNNLSTMRegressor, self).__init__(batch_size, epochs)
        self.filters = filters
        self.pooling = pooling
        self.pool_size = pool_size
        self.dropout = dropout
        self.lstm_units = lstm_units
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.trainable = trainable
        self.initialize()

    def initialize(self):
        if self.hidden_dims is None:
            self.hidden_dims = []
        else:
            self.hidden_dims = list(self.hidden_dims)
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size]

    def preprocess(self, X, y=None):
        if not hasattr(self, 'embedding_matrix_'):
            self.embedding_matrix_ = X['embedding_matrix']
        if not hasattr(self, 'vocab_size_'):
            self.vocab_size_ = self.embedding_matrix_.shape[0]
        if not hasattr(self, 'sequence_length_'):
            self.sequence_length_ = X['tokens'].shape[1]
        X = X['tokens']
        return X, y

    def build_model(self, X, y):
        """
        Compiles the CNN LSTM model

        :return: compiled classifier
        """
        ipt = Input(shape=(self.sequence_length_,), sparse=False, dtype='int32')
        opt = ipt

        # Embedding layer (Extracts embedding from embedding matrix according to input index sequence)
        e0 = Embedding(self.vocab_size_, self.embedding_matrix_.shape[1], weights=[self.embedding_matrix_],
                       input_length=self.sequence_length_, trainable=self.trainable)
        opt = e0(opt)

        # Dropout layer
        if self.dropout is not None:
            d0 = Dropout(self.dropout)
            opt = d0(opt)

        p1 = []
        for ks in self.kernel_size:
            # Convolution1D Layer
            c0 = Conv1D(self.filters, ks, padding='valid', activation='relu', strides=1)(opt)
            # Pooling layer: don't pool if self.pooling is not defined
            p0 = {'max': MaxPooling1D(pool_size=self.pool_size)(c0),
                  'avg': AveragePooling1D(pool_size=self.pool_size)(c0)}.get(self.pooling, c0)
            p1.append(p0)
        if len(p1) > 1:
            opt = keras.layers.concatenate(p1, axis=1)
        else:
            opt = p1[0]

        lstm = LSTM(units=self.lstm_units)
        opt = lstm(opt)

        # Hidden layers
        for dim in self.hidden_dims:
            l0 = Dense(dim, activation='relu')
            opt = l0(opt)

        # Output layer with sigmoid activation:
        opt = Dense(1, activation='sigmoid')(opt)

        model = Model(inputs=ipt, outputs=opt)

        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class LSTMCNNRegressor(KerasRegressor):
    def __init__(self, filters=250, kernel_size=3, pooling=None, dropout=None, lstm_units=300,
                 hidden_dims=None, trainable=False, batch_size=32, epochs=15):
        """
        Initializes the classifier

        :param filters: Number output of filters in the convolution.
        :param kernel_size: Length of the 1D convolution window.
        :param pooling: Type of pooling; pooling is not done if None.
        :param dropout: dropout after cnn layer.
        :param lstm_units: number of lstm units
        :param hidden_dims: List of dimensionality of the output space of the dense layers. None will be default to [250, 50].
        :param trainable: Whether cnn is trainable or not
        :param batch_size: Number of samples per gradient update. If unspecified, it will default to 32.
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        """
        super(LSTMCNNRegressor, self).__init__(batch_size, epochs)
        self.filters = filters
        self.pooling = pooling
        self.dropout = dropout
        self.lstm_units = lstm_units
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.trainable = trainable
        self.initialize()

    def initialize(self):
        if self.hidden_dims is None:
            self.hidden_dims = []
        else:
            self.hidden_dims = list(self.hidden_dims)
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size]

    def preprocess(self, X, y=None):
        if not hasattr(self, 'embedding_matrix_'):
            self.embedding_matrix_ = X['embedding_matrix']
        if not hasattr(self, 'vocab_size_'):
            self.vocab_size_ = self.embedding_matrix_.shape[0]
        if not hasattr(self, 'sequence_length_'):
            self.sequence_length_ = X['tokens'].shape[1]
        if not hasattr(self, 'num_categories_'):
            self.num_categories_ = len(np.unique(y))
        if y is not None:
            y = to_categorical(y)
        X = X['tokens']
        return X, y

    def build_model(self, X, y):
        """
        Compiles the LSTM CNN model

        :return: compiled classifier
        """
        ipt = Input(shape=(self.sequence_length_,), sparse=False, dtype='int32')
        opt = ipt

        # Embedding layer (Extracts embedding from embedding matrix according to input index sequence)
        e0 = Embedding(self.vocab_size_, self.embedding_matrix_.shape[1], weights=[self.embedding_matrix_],
                       input_length=self.sequence_length_, trainable=self.trainable)
        opt = e0(opt)

        lstm = LSTM(units=self.lstm_units, return_sequences=True)
        opt = lstm(opt)

        p1 = []
        for ks in self.kernel_size:
            # Convolution1D Layer
            c0 = Conv1D(self.filters, ks, padding='valid', activation='relu', strides=1)(opt)
            # Pooling layer:
            p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(self.pooling, c0)
            p1.append(p0)
        if len(p1) > 1:
            opt = keras.layers.concatenate(p1, axis=1)
        else:
            opt = p1[0]

        # Dropout layers
        if self.dropout is not None:
            d0 = Dropout(self.dropout)
            opt = d0(opt)

        # Hidden layers
        for dim in self.hidden_dims:
            l0 = Dense(dim, activation='relu')
            opt = l0(opt)

        # Output layer with sigmoid activation:
        opt = Dense(1, activation='sigmoid')(opt)

        model = Model(inputs=ipt, outputs=opt)

        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
