# -*- coding: utf-8 -*-
from classes.methods import naive,ets,arima,rf,mlp,cnn,lstm,cnn_lstm,convlstm,twodlstm

class forecasting_method():

  def __init__(self, parameters): 
    self.parameters = parameters
    self.method = self.parameters.get('method')
    self.method_parameters = self.parameters.get('method_parameters')

    if self.method == 'Naive':
      self.method_instance = naive.method(self.method_parameters)
    if self.method == 'ETS':
      self.method_instance = ets.method(self.method_parameters)
    if self.method == 'ARIMA':
      self.method_instance = arima.method(self.method_parameters)
    if self.method == 'RF':
      self.method_instance = rf.method(self.method_parameters)
    if self.method == 'MLP':
      self.method_instance = mlp.method(self.method_parameters)
    if self.method == 'CNN':
      self.method_instance = cnn.method(self.method_parameters)
    if self.method == 'LSTM':
      self.method_instance = lstm.method(self.method_parameters)
    if self.method == 'CNN-LSTM':
      self.method_instance = cnn_lstm.method(self.method_parameters)
    if self.method == 'ConvLSTM':
      self.method_instance = convlstm.method(self.method_parameters)
    if self.method == 'twoDLSTM':
      self.method_instance = twodlstm.method(self.method_parameters)

  #main 
  def get_model(self):
    return self.method_instance
  
  def get_reshaped_data(self, data_to_reshape, outcome_variable, time_variable):
    return self.method_instance.reshape_data_to_predict(data_to_reshape, outcome_variable, time_variable)
  
  def get_updated_reshaped_data(self, reshaped_data, new_observation):
    return self.method_instance.update_reshaped_data(reshaped_data, new_observation)