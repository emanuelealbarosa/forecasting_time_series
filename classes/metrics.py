from math import sqrt


class evaluator:

    def __init__(self, metrics_parameter):
        self.metrics_type = metrics_parameter.get('type')

    def score(self, predictions, values):
        score = None

        if self.metrics_type == 'rmse':
            score = sqrt(
                sum([(prediction - value) ** 2 for prediction, value in zip(predictions, values)]) / len(values))

        return score
