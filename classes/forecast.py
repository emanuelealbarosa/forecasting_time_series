import classes.method as method
import classes.metrics as metrics
import classes.processing as processing


class forecast_agent:

    def __init__(self, data, parameters):
        self.data = data
        self.parameters = parameters
        self.method_parameters = self.parameters.get('method_parameters')
        self.processing_parameters = self.parameters.get('processing_parameters')
        self.forecasting_period = self.parameters.get('general_parameters').get('forecasting_period')
        self.outcome_variable = self.parameters.get('general_parameters').get('outcome_variable')
        self.time_sequence_variable = self.parameters.get('general_parameters').get('time_sequence_variable')
        self.metrics = self.parameters.get('general_parameters').get('metrics')
        self.method = method.forecasting_method(self.parameters)
        self.model = self.method.get_model()

        self.training_set = None
        self.validation_set = None
        self.processer = None
        self.outcome_data = None
        self.predictions = []

        self.forecast_score = None

    # main methods
    def evaluate(self):
        self.training_set, self.validation_set = self.split_train_validation_sets()
        self.processer = processing.data_processer(self.training_set, self.parameters)
        self.processer.transform()
        self.training_set = self.processer.data
        self.outcome_data = self.method.get_reshaped_data(self.training_set,
                                                          self.outcome_variable,
                                                          self.time_sequence_variable)
        self.predict_steps_forward()
        self.detransform_predictions()
        self.update_forecast_score()

    # subsidiary methods
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
            self.forecast_score = metrics.evaluator(self.metrics).score(self.predictions, validation_values)
        else:
            self.forecast_score = None

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
