# -*- coding: utf-8 -*-
"""naive
"""

import numpy 

class method():

  def __init__(self, method_parameters):
    self.average_type = method_parameters.get('average_type')
    self.length = method_parameters.get('predictors_length')
    self.offset = method_parameters.get('predictors_offset')
  
  #main
  def reshape_data_to_predict(self, data_to_reshape, outcome_variable, time_variable):
    data = [value for value in data_to_reshape[outcome_variable].values]
    return data

  def predict(self, values):
    prediction = None
    if self.check_offset_times_length_ok(values):
      prediction = self.get_naive_prediction(values)
    return prediction

  def update_reshaped_data(self, reshaped_data, new_observation):
    reshaped_data.append(new_observation)
    return reshaped_data

  #subsidiary
  def check_offset_times_length_ok(self, values):
    return self.length*self.offset <= len(values)
  
  def get_naive_prediction(self, values):
    prediction = None
    if self.average_type == 'persist' or self.length == 1:
      prediction = self.get_persist_value(values)
    else:
      values_subset = self.get_values_to_average(values)
      prediction = self.average_values(values_subset)
    return prediction
  
  def get_persist_value(self, values):
    return values[-self.length*self.offset]

  def get_values_to_average(self, values):
    values_subset = []
    if self.offset == 1:  
        values_subset = values[-self.length:]
    else:
        for i in range(1, self.length + 1):
          ix = i*self.offset
          values_subset.append(values[-ix])
    return values_subset

  def average_values(self, values):
    avg = None
    if self.average_type == 'mean':
      avg = numpy.nanmean(values)
    else:
      if self.average_type == 'median':
        avg = numpy.nanmedian(values)
      else:
        avg = None
    return avg