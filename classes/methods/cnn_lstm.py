# -*- coding: utf-8 -*-
"""CNN-LSTM
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, LSTM, Dense, TimeDistributed)


class method():

    def __init__(self, method_parameters):
        self.number_input_lags = method_parameters.get('number_input_lags')
        self.number_subsequences = method_parameters.get('number_subsequences')
        self.number_filters = method_parameters.get('number_filters')
        self.kernel_size = method_parameters.get('kernel_size')
        self.number_lstm_units = method_parameters.get('number_lstm_units')
        self.number_epochs = method_parameters.get('number_epochs')
        self.batch_size = method_parameters.get('batch_size')
        self.subsequence_length = None
        if self.number_subsequences:
            self.subsequence_length = self.number_input_lags // self.number_subsequences

    # main
    def reshape_data_to_predict(self, data_to_reshape, outcome_variable, time_variable):
        return numpy.array([value for value in data_to_reshape[outcome_variable].values], dtype=float)

    def predict(self, values):
        prediction = None
        if self.check_length_ok(values):
            try:
                X, y = self.make_windows(values)
                X = X.reshape((X.shape[0], self.number_subsequences, self.subsequence_length, 1))
                model = self.build_model()
                model.fit(X, y, epochs=self.number_epochs, batch_size=self.batch_size, verbose=0)
                X_input = values[-self.number_input_lags:].reshape(
                    1, self.number_subsequences, self.subsequence_length, 1)
                prediction = float(model.predict(X_input, verbose=0)[0][0])
            except Exception:
                prediction = None
        return prediction

    def update_reshaped_data(self, reshaped_data, new_observation):
        return numpy.append(reshaped_data, new_observation)

    # subsidiary
    def check_length_ok(self, values):
        return (self.subsequence_length is not None and self.subsequence_length >= self.kernel_size
                and self.number_input_lags % self.number_subsequences == 0
                and len(values) > self.number_input_lags)

    def make_windows(self, values):
        X, y = [], []
        for i in range(self.number_input_lags, len(values)):
            X.append(values[i - self.number_input_lags:i])
            y.append(values[i])
        return numpy.array(X), numpy.array(y)

    def build_model(self):
        model = Sequential()
        model.add(TimeDistributed(
            Conv1D(filters=self.number_filters, kernel_size=self.kernel_size, activation='relu'),
            input_shape=(self.number_subsequences, self.subsequence_length, 1)))
        pool_size = 2 if (self.subsequence_length - self.kernel_size + 1) >= 2 else 1
        model.add(TimeDistributed(MaxPooling1D(pool_size=pool_size)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(self.number_lstm_units, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
