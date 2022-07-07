import numpy
from classes import grid_builder as grid_builder
import classes.validation as validation


class search_agent:

  def __init__(self, data, method, general_parameters, method_and_processing_parameter_ranges):
    self.data = data
    self.method = method
    self.general_parameters = general_parameters
    self.method_and_processing_parameter_ranges = method_and_processing_parameter_ranges
    self.processing_parameters_number = self.general_parameters.get('processing_parameters_number')
    self.outcome_variable = self.general_parameters.get('outcome_variable')
    self.time_variable = self.general_parameters.get('time_variable')
    self.forecasting_period = self.general_parameters.get('forecasting_period')
    self.validation_minimum_training_split = self.general_parameters.get('validation_minimum_training_split')
    self.validation_maximum_number_splits = self.general_parameters.get('validation_maximum_number_of_splits')

    self.grid_of_parameter_combinations = grid_builder.builder(
      self.method, self.method_and_processing_parameter_ranges
    ).get_grid_of_parameter_combinations()
    self.parameters = {'method': self.method,
                       'general_parameters': self.general_parameters,
                       'method_parameters': None,
                       'processing_parameters': None
                       }
    self.search_outputs = []
    self.search_invalid_outputs = []

# main methods
  def identify_best_combination_of_parameters(self):
    self.search_through_combinations_of_parameters()
    self.store_invalid_outcomes()
    self.clean_search_outputs()
    self.sort_search_outputs()

# subsidiary methods
  def search_through_combinations_of_parameters(self):
    for combination in self.grid_of_parameter_combinations:
      self.update_parameters(combination)
      self.search_and_evaluate_single_combination_of_parameters()
  
  def store_invalid_outcomes(self):
    self.search_invalid_outputs = [output for output in self.search_outputs if output[1] is None or
                                   numpy.isnan(output[1]) or numpy.isinf(output[1])]

  def clean_search_outputs(self):
    self.search_outputs = [output for output in self.search_outputs if output[1] is not None
                           and not numpy.isnan(output[1]) and not numpy.isinf(output[1])]
  
  def sort_search_outputs(self):
    self.search_outputs.sort(key=lambda x: float(x[1]))
  
  def update_parameters(self, combination):
    combination_list = list(combination)
    method_parameters = {}
    processing_parameters = {}
    for key in combination_list[:-self.processing_parameters_number]:
      method_parameters[key] = combination[key]
    for key in combination_list[-self.processing_parameters_number:]:
      processing_parameters[key] = combination[key]
    self.parameters['method_parameters'] = method_parameters
    self.parameters['processing_parameters'] = processing_parameters

  def search_and_evaluate_single_combination_of_parameters(self):
    validator = validation.validation_agent(self.data, self.parameters)
    validator.validate()
    self.search_outputs.append([validator.key, validator.error_overall_score, validator.model_instance])
