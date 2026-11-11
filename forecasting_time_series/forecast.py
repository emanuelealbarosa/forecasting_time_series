from . import method
from . import metrics
from . import processing


class forecast_agent:

    def __init__(self, data, parameters):
        self.data = data
        self.parameters = parameters
        self.method_parameters = self.parameters.get('method_parameters')
        self.processing_parameters = self.parameters.get('processing_parameters')
        self.forecasting_period = self.parameters.get('general_parameters').get('forecasting_period')
        self.outcome_variable = self.parameters.get('general_parameters').get('outcome_variable')
        self.time_sequence_variable = self.parameters.get('general_parameters').get('time_sequence_variable')
        self.time_variable = self.parameters.get('general_parameters').get('time_variable')
        self.exog_variables = list(self.parameters.get('general_parameters').get('exog_variables') or [])
        # the exogenous variables whose values over the forecast horizon are known in advance (the rest
        # are used only through their past values, so their horizon values never reach the method)
        known = self.parameters.get('general_parameters').get('known_future_exog')
        self.known_future_exog = list(self.exog_variables if known is None else known)
        self.metrics = self.parameters.get('general_parameters').get('metrics')
        self.method = method.forecasting_method(self.parameters)
        self.model = self.method.get_model()

        self.training_set = None
        self.validation_set = None
        self.future_exog = None   # exogenous values over the forecast horizon (known in advance)
        self.processer = None
        self.outcome_data = None
        self.predictions = []

        self.forecast_score = None
        self.failure_reason = None  # why no score could be computed, if so

    # main methods
    def evaluate(self):
        """Hold out the last `forecasting_period` rows, forecast them and score the forecast."""
        self.training_set, self.validation_set = self.split_train_validation_sets()
        if self.known_future_exog:
            # the held-out rows' values of the variables known in advance; their outcome, and the
            # values of variables known only up to the past, stay hidden
            self.future_exog = self.validation_set[self.known_future_exog].reset_index(drop=True)
        try:
            self.fit_and_predict()
        except ValueError as error:  # e.g. log of values <= 0, or a series with no spread
            self.forecast_score = None
            self.failure_reason = str(error)
            return
        self.update_forecast_score()

    def forecast(self, future_exog=None):
        """Use all of `data` and forecast `forecasting_period` steps beyond its end.
        future_exog: the exogenous variables' values over those steps (needed if the method uses any)."""
        self.training_set = self.data.copy(deep=True)
        if self.known_future_exog:
            if future_exog is None or len(future_exog) < self.forecasting_period:
                raise ValueError(f'future_exog with at least {self.forecasting_period} rows of '
                                 f'{self.known_future_exog} is needed to forecast')
            self.future_exog = future_exog[self.known_future_exog].iloc[:self.forecasting_period] \
                .reset_index(drop=True)
        self.fit_and_predict()
        return self.predictions

    # subsidiary methods
    def fit_and_predict(self):
        self.processer = processing.data_processer(self.training_set, self.parameters)
        self.processer.transform()
        self.training_set = self.processer.data
        if hasattr(self.model, 'predict_horizon'):
            # direct multi-step method (e.g. LEAR): the whole horizon in one go, from the training data
            # (with the exogenous variables' past values) and the exogenous values over the horizon
            self.predictions = [float(value) for value in self.model.predict_horizon(
                self.training_set, self.outcome_variable, self.time_variable, self.forecasting_period,
                exog_variables=self.exog_variables, future_exog=self.future_exog)]
            if not all(self.check_prediction(value) for value in self.predictions):
                raise ValueError('the method produced a forecast that is not a finite number')
        else:
            if self.exog_variables:
                raise ValueError(f'{self.parameters.get("method")} cannot use exogenous variables')
            # recursive method (e.g. Naive): one step at a time, each forecast fed back in
            self.outcome_data = self.method.get_reshaped_data(self.training_set,
                                                              self.outcome_variable,
                                                              self.time_sequence_variable)
            self.predict_steps_forward()
        self.detransform_predictions()

    def split_train_validation_sets(self):
        return self.data.copy(deep=True)[:-self.forecasting_period], \
               self.data.copy(deep=True)[-self.forecasting_period:]

    def extract_outcome_values(self, data):
        return [value for value in data[self.outcome_variable].values]

    def predict_steps_forward(self):
        prediction = 0
        step = 1
        while step <= self.forecasting_period and self.check_prediction(prediction):
            prediction = self.predict_if_not_none(self.outcome_data)
            if self.check_prediction(prediction):
                self.predictions.append(prediction)
                self.outcome_data = self.method.get_updated_reshaped_data(self.outcome_data, prediction)
            step = step + 1

    def detransform_predictions(self):
        self.processer.detransform(self.predictions)
        self.predictions = self.processer.new_detransformed_data

    def update_forecast_score(self):
        if len(self.predictions) > 0:
            validation_values = [self.validation_set[self.outcome_variable].values[i]
                                 for i in range(len(self.predictions))]
            try:
                self.forecast_score = metrics.evaluator(self.metrics).score(self.predictions, validation_values)
            except ValueError as error:  # e.g. MAPE with an actual value of 0
                self.forecast_score = None
                self.failure_reason = str(error)
        else:
            self.forecast_score = None
            self.failure_reason = 'the method produced no forecast (e.g. too little history for its settings)'

    def predict_if_not_none(self, values):
        result = None
        if self.model is not None:
            result = self.model.predict(values)
        return result

    def check_prediction(self, prediction):
        status = None
        if prediction is not None:
            status = prediction == prediction and prediction != float('inf')
        return status
