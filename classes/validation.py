import numpy
import classes.forecast as forecast


class validation_agent:

    def __init__(self, data, parameters):
        self.data = data
        self.parameters = parameters
        self.method_parameters = self.parameters.get('method_parameters')
        self.outcome_variable = self.parameters.get('general_parameters').get('outcome_variable')
        self.time_variable = self.parameters.get('general_parameters').get('time_variable')
        self.forecasting_period = self.parameters.get('general_parameters').get('forecasting_period')
        self.validation_minimum_training_split = self.parameters.get('general_parameters').get(
            'validation_minimum_training_split')
        self.validation_maximum_number_splits = self.parameters.get('general_parameters').get(
            'validation_maximum_number_of_splits')
        self.validation_number_of_splits = self.validation_maximum_number_splits
        self.validation_splits_indices = []

        self.model_instance = None
        self.errors = []
        self.error_overall_score = None
        self.key = str(self.parameters)

    # main method
    def validate(self):
        self.update_validation_steps_number()
        self.update_validation_splits()
        self.forecast_through_splits()
        self.clean_invalid_error_values()
        self.evaluate_errors()

    # subsidiary methods
    def update_validation_steps_number(self):
        while not self.check_length_of_validation_period() and self.validation_number_of_splits > 1:
            self.validation_number_of_splits = self.validation_number_of_splits - 1

    def update_validation_splits(self):
        for split in range(self.validation_number_of_splits):
            self.validation_splits_indices.append(len(self.data) - split * self.forecasting_period)
        self.validation_splits_indices = self.validation_splits_indices[::-1]  # reverse order

    def forecast_through_splits(self):
        for index in self.validation_splits_indices:
            forecast_single_split_output = self.forecast_single_split(index)
            self.errors.append(forecast_single_split_output[0])
        self.model_instance = forecast_single_split_output[1]

    def clean_invalid_error_values(self):
        self.errors = [error for error in self.errors if
                       error is not None and not numpy.isnan(error) and not numpy.isinf(error)]

    def evaluate_errors(self):
        self.error_overall_score = numpy.mean(self.errors)

    def check_length_of_validation_period(self):
        status = (len(self.data) - self.validation_number_of_splits * self.forecasting_period) < len(
            self.data) * self.validation_minimum_training_split
        return status

    def forecast_single_split(self, index):
        data_split = self.data[:index]
        if self.check_nas_in_training(data_split):
            forecaster = forecast.forecast_agent(data_split, self.parameters)
            forecaster.evaluate()
            forecast_output = [forecaster.forecast_score, forecaster.model]
        else:
            forecast_output = [None, None]
        return forecast_output

    def check_nas_in_training(self, data):
        status = numpy.sum(numpy.isnan(data[self.outcome_variable]) <= self.forecasting_period + 2)
        return status
