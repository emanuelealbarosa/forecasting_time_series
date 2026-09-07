# -*- coding: utf-8 -*-
"""LSTM
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense


class method():

    def __init__(self, method_parameters):
        self.number_input_lags = method_parameters.get('number_input_lags')
        self.number_lstm_units = method_parameters.get('number_lstm_units')
        self.number_epochs = method_parameters.get('number_epochs')
        self.batch_size = method_parameters.get('batch_size')

    # main
    def reshape_data_to_predict(self, data_to_reshape, outcome_variable, time_variable):
        return numpy.array([value for value in data_to_reshape[outcome_variable].values], dtype=float)

    def predict(self, values):
        prediction = None
        if self.check_length_ok(values):
            try:
                X, y = self.make_windows(values)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model = self.build_model()
                model.fit(X, y, epochs=self.number_epochs, batch_size=self.batch_size, verbose=0)
                X_input = values[-self.number_input_lags:].reshape(1, self.number_input_lags, 1)
                prediction = float(model.predict(X_input, verbose=0)[0][0])
            except Exception:
                prediction = None
        return prediction

    def update_reshaped_data(self, reshaped_data, new_observation):
        return numpy.append(reshaped_data, new_observation)

    # subsidiary
    def check_length_ok(self, values):
        return len(values) > self.number_input_lags

    def make_windows(self, values):
        X, y = [], []
        for i in range(self.number_input_lags, len(values)):
            X.append(values[i - self.number_input_lags:i])
            y.append(values[i])
        return numpy.array(X), numpy.array(y)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.number_lstm_units, activation='relu', input_shape=(self.number_input_lags, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
