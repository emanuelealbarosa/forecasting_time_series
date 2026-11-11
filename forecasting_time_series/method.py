import importlib

# The module of each method, imported on first use: the neural methods need torch, which is an optional
# dependency (pip install forecasting-time-series[neural]), so it is only imported when MLP or LSTM is used.
#
# Each method module defines required_history(method_parameters): how many training observations (after
# preprocessing) it needs to make its first forecast, and a `method` class.
# Recursive methods (Naive) forecast one step at a time through predict / update_reshaped_data.
# Direct methods (LEAR, MLP, LSTM, RF) define predict_horizon(data, outcome, time_variable, horizon) instead,
# and forecast the whole horizon at once; forecast_agent uses whichever the method has.
METHOD_MODULES = {'Naive': 'naive',
                  'LEAR': 'lear',
                  'MLP': 'mlp',
                  'LSTM': 'lstm',
                  'RF': 'random_forest'}

# the optional package each method needs, for a clearer error when it is missing
OPTIONAL_DEPENDENCIES = {'MLP': ('torch', 'neural'), 'LSTM': ('torch', 'neural')}


def method_module(method_name):
    """The module implementing `method_name` (e.g. 'LEAR')."""
    if method_name not in METHOD_MODULES:
        raise ValueError(f'unknown method {method_name!r}; choose from {list(METHOD_MODULES)}')
    try:
        return importlib.import_module(f'.methods.{METHOD_MODULES[method_name]}', __package__)
    except ModuleNotFoundError as error:
        package, extra = OPTIONAL_DEPENDENCIES.get(method_name, (None, None))
        if package is not None and error.name == package:
            raise ModuleNotFoundError(f'the {method_name} method needs {package}: '
                                      f'pip install "forecasting-time-series[{extra}]"', name=package) from error
        raise


def required_history(method_name, method_parameters):
    return method_module(method_name).required_history(method_parameters)


class forecasting_method():

    def __init__(self, parameters):
        self.parameters = parameters
        self.method = self.parameters.get('method')
        self.method_parameters = self.parameters.get('method_parameters')
        self.method_instance = method_module(self.method).method(self.method_parameters)

    # main
    def get_model(self):
        return self.method_instance

    def get_reshaped_data(self, data_to_reshape, outcome_variable, time_variable):
        return self.method_instance.reshape_data_to_predict(data_to_reshape, outcome_variable, time_variable)

    def get_updated_reshaped_data(self, reshaped_data, new_observation):
        return self.method_instance.update_reshaped_data(reshaped_data, new_observation)
