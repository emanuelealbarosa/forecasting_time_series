import pandas as pd
import classes.grid_search as grid_search

outcome_variable = 'crossings'
time_variable = 'date'
time_sequence_variable = 'date_sequence'
forecasting_period = 7
metrics = {'type': 'rmse'}
validation_minimum_training_split = 0.7
validation_maximum_number_of_steps = 7
processing_parameters_number = 3

data = pd.DataFrame([[i, 1, 1] for i in range(1, 1000, 2)],
                    columns=[outcome_variable, time_variable, time_sequence_variable])
for i in data.index:
  data['crossings'].values[i] = data['crossings'].values[i]*i
forecasting_method_name = 'Naive'
differencing_order_range = [0, 1]
power_exponent_range = [1, 0.5]
is_normalized_range = [False, True]


forecasting_method_name = 'ARIMA'


"""Naive"""

if forecasting_method_name == 'Naive':
    predictors_maximum_length_range = [10, 20]
    predictors_offset_range = [0, 5]
    average_type_range = ['mean', 'median']
    method_parameter_ranges = {'predictors_maximum_length_range': predictors_maximum_length_range,
                               'predictors_offset_range': predictors_offset_range,
                               'average_type_range': average_type_range}

"""ETS"""

if forecasting_method_name == 'ETS':
    trend_component_range = ['add', 'mul', None]
    is_trend_damped_range = [True, False]
    seasonal_component_range = ['add', 'mul', None]
    seasonal_period_range = [0, 5, 10]
    method_parameter_ranges = {'trend_component_range': trend_component_range,
                               'is_trend_damped_range': is_trend_damped_range,
                               'seasonal_component_range': seasonal_component_range,
                               'seasonal_period_range': seasonal_period_range}

"""ARIMA"""

if forecasting_method_name == 'ARIMA':
    ar_terms_range = [5, 10]
    seasonal_ar_terms_range = [0]
    differencing_terms_range = [1, 2]
    seasonal_differencing_terms_range = [0]
    ma_terms_range = [5, 10]
    seasonal_ma_terms_range = [0]
    seasonal_period_range = [0]
    trend_range = ['n']
    method_parameter_ranges = {'ar_terms_range': ar_terms_range,
                               'seasonal_ar_terms_range': seasonal_ar_terms_range,
                               'differencing_terms_range': differencing_terms_range,
                               'seasonal_differencing_terms_range': seasonal_differencing_terms_range,
                               'ma_terms_range': ma_terms_range,
                               'seasonal_ma_terms_range': seasonal_ma_terms_range,
                               'seasonal_period_range': seasonal_period_range,
                               'trend_range': trend_range}

"""RF"""

if forecasting_method_name == 'RF':
    number_input_lags_range = [10, 20]
    number_output_leads_range = [1]
    minimum_samples_leaf_range = [5]
    maximum_number_features_range = [5, 10]
    number_trees_range = [10]
    method_parameter_ranges = {'number_input_lags_range': number_input_lags_range,
                               'number_output_leads_range': number_output_leads_range,
                               'minimum_samples_leaf_range': minimum_samples_leaf_range,
                               'maximum_number_features_range': maximum_number_features_range,
                               'number_trees_range': number_trees_range}


######
general_parameters = {'outcome_variable': outcome_variable,
                      'time_variable': time_variable,
                      'time_sequence_variable': time_sequence_variable,
                      'forecasting_period': forecasting_period,
                      'metrics': metrics,
                      'validation_minimum_training_split': validation_minimum_training_split,
                      'validation_maximum_number_of_splits': validation_maximum_number_of_steps,
                      'processing_parameters_number': processing_parameters_number
                      }
processing_parameter_ranges = {'differencing_order_range': differencing_order_range,
                               'power_exponent_range': power_exponent_range,
                               'is_normalized_range': is_normalized_range
                               }
method_and_processing_parameter_ranges = {'method_parameter_ranges': method_parameter_ranges,
                                          'processing_parameter_ranges': processing_parameter_ranges
                                          }


####
trial = grid_search.search_agent(data,
                                 forecasting_method_name,
                                 general_parameters,
                                 method_and_processing_parameter_ranges
                                 )
trial.identify_best_combination_of_parameters()

for i in range(5):
    print('Model ' + str(i + 1))
    print(trial.search_outputs[i][0])
    print(metrics.get('type') + ': ' + str(trial.search_outputs[i][1]))
    print(' ')
