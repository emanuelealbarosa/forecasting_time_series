import time

import numpy
from . import forecast


def plan_splits(number_of_rows, forecasting_period, maximum_number_of_splits, minimum_training_share,
                required_history=0):
    """How many walk-forward splits to use, and whether the earliest one has enough history.

    Starts from the maximum and removes splits until the training part of the earliest
    split (the rows before its test window) is at least `minimum_training_share` of the
    data and at least `required_history` rows. Never goes below 1 split; if even 1 split
    is too short, `meets_minimum` is False."""
    minimum_training_rows = max(number_of_rows * minimum_training_share, required_history)

    def earliest_training_rows(splits):
        return number_of_rows - splits * forecasting_period

    splits = maximum_number_of_splits
    while splits > 1 and earliest_training_rows(splits) < minimum_training_rows:
        splits -= 1
    meets_minimum = earliest_training_rows(splits) >= minimum_training_rows
    return splits, meets_minimum


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
        # set by the grid search so that every combination uses the same splits; None → plan here
        self.validation_number_of_splits = self.parameters.get('general_parameters').get(
            'validation_number_of_splits')
        self.validation_splits_indices = []

        self.model_instance = None
        self.errors = []               # one score per split that succeeded
        self.failure_reasons = []      # one reason per split that failed
        self.split_predictions = []    # forecasts of each split that succeeded (used for ensembles)
        self.split_actuals = []        # the actual values they were scored against
        self.split_seconds = []        # wall seconds to fit and forecast each split that succeeded
        self.error_overall_score = None

    # main method
    def validate(self):
        self.update_validation_steps_number()
        self.update_validation_splits()
        self.forecast_through_splits()
        self.clean_invalid_error_values()
        self.evaluate_errors()

    # subsidiary methods
    def update_validation_steps_number(self):
        if self.validation_number_of_splits is not None:
            return
        self.validation_number_of_splits, _ = plan_splits(len(self.data),
                                                          self.forecasting_period,
                                                          self.validation_maximum_number_splits,
                                                          self.validation_minimum_training_split)

    def update_validation_splits(self):
        for split in range(self.validation_number_of_splits):
            self.validation_splits_indices.append(len(self.data) - split * self.forecasting_period)
        self.validation_splits_indices = self.validation_splits_indices[::-1]  # reverse order

    def forecast_through_splits(self):
        for index in self.validation_splits_indices:
            started = time.perf_counter()
            score, model, failure_reason, predictions, actual = self.forecast_single_split(index)
            seconds = time.perf_counter() - started
            self.errors.append(score)
            if failure_reason is not None:
                self.failure_reasons.append(failure_reason)
            else:  # success: same order as the scores kept by clean_invalid_error_values
                self.split_predictions.append(predictions)
                self.split_actuals.append(actual)
                self.split_seconds.append(seconds)
            self.model_instance = model

    def clean_invalid_error_values(self):
        self.errors = [error for error in self.errors if
                       error is not None and not numpy.isnan(error) and not numpy.isinf(error)]

    def evaluate_errors(self):
        self.error_overall_score = numpy.mean(self.errors) if self.errors else None

    def forecast_single_split(self, index):
        # the split's test window is the last forecasting_period rows before `index`; its training
        # data is everything before that (expanding window) or the last `training_window` rows (fixed)
        training_window = self.parameters.get('training_window')
        start = 0 if training_window is None else max(0, index - self.forecasting_period - training_window)
        data_split = self.data[start:index]
        if self.has_few_enough_missing_values(data_split):
            forecaster = forecast.forecast_agent(data_split, self.parameters)
            forecaster.evaluate()
            failure_reason = forecaster.failure_reason
            if failure_reason is None and not self.is_valid_score(forecaster.forecast_score):
                failure_reason = 'the forecast was not a finite number'
            predictions = [float(value) for value in forecaster.predictions]
            actual = [float(value) for value in
                      forecaster.validation_set[self.outcome_variable].values[:len(predictions)]]
            return forecaster.forecast_score, forecaster.model, failure_reason, predictions, actual
        return None, None, 'too many missing values in the training data', None, None

    @staticmethod
    def is_valid_score(score):
        return score is not None and not numpy.isnan(score) and not numpy.isinf(score)

    def has_few_enough_missing_values(self, data):
        """At most forecasting_period + 2 missing outcome values are allowed in a split."""
        number_missing = int(numpy.sum(numpy.isnan(data[self.outcome_variable])))
        return number_missing <= self.forecasting_period + 2
