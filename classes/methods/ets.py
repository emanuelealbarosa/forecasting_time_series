import numpy
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class method:

  def __init__(self, method_parameters):
    self.trend_component = method_parameters.get('trend_component')
    self.is_trend_damped = method_parameters.get('is_trend_damped')
    self.seasonal_component = method_parameters.get('seasonal_component')
    self.seasonal_period = method_parameters.get('seasonal_period')

# main method
  def reshape_data_to_predict(self, data_to_reshape, outcome_variable):
    data = numpy.array([value for value in data_to_reshape[outcome_variable].values])
    return data

  def predict(self, values):
    values = self.impute_non_positive_values(values)
    prediction = self.get_prediction(values)
    return prediction
    
  def update_reshaped_data(self, reshaped_data, new_observation):
    return numpy.append(reshaped_data, new_observation)

# subsidiary methods
  def impute_non_positive_values(self, values):
    values[values == 0] = 0.0001
    return values
  
  def get_prediction(self, values):
    try:
      model = ExponentialSmoothing(values,
                                   trend=self.trend_component,
                                   damped_trend=self.is_trend_damped,
                                   seasonal=self.seasonal_component,
                                   seasonal_periods=self.seasonal_period)
      model_fit = model.fit(optimized=True)
      prediction = model_fit.predict(len(values), len(values))
    except:
      prediction = None
    return prediction
