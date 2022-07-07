import numpy


class data_processer:

    def __init__(self, data, parameters):
        self.data = data.copy(deep=True)
        self.parameters = parameters
        self.outcome_variable = self.parameters.get('general_parameters').get('outcome_variable')
        self.time_sequence_variable = self.parameters.get('general_parameters').get('time_sequence_variable')
        self.differencing_order = self.parameters.get('processing_parameters').get('differencing_order')
        self.power_exponent = self.parameters.get('processing_parameters').get('power_exponent')
        self.is_normalized = self.parameters.get('processing_parameters').get('is_normalized')
        self.differencing_correction_values = self.data[self.outcome_variable].shift(self.differencing_order)
        self.center_value = 0.0
        self.scale_value = 1.0
        self.new_transformed_data = None  # this allows us to store predictions made on transformed data
        self.new_detransformed_data = None
        self.new_differencing_correction_values = []

    # Main methods
    def transform(self):
        self.difference_data_if_needed()
        self.power_transform_data()
        self.update_center_value_if_needed()
        self.update_scale_value_if_needed()
        self.normalize_data()

    def detransform(self, new_data_to_detransform):
        self.new_transformed_data = new_data_to_detransform  # store
        self.new_detransformed_data = new_data_to_detransform
        self.denormalize_data()
        self.depower_transform_data()
        self.dedifference_data_if_needed()

    # Subsidiary methods
    def difference_data_if_needed(self):
        if self.differencing_order > 0:
            self.data[self.outcome_variable] = \
                self.data.groupby(self.time_sequence_variable)[self.outcome_variable].diff(self.differencing_order)
            self.data = self.data.dropna()

    def power_transform_data(self):
        self.data[self.outcome_variable] = self.power_transform(self.data[self.outcome_variable].values)

    def power_transform(self, values):
        result = []
        for value in values:
            if value >= 0:
                power = value ** self.power_exponent
            else:
                power = -(abs(value) ** self.power_exponent)
            result.append(power)
        return result

    def update_center_value_if_needed(self):
        if self.is_normalized:
            self.center_value = numpy.nanmin(self.data[self.outcome_variable])

    def update_scale_value_if_needed(self):
        if self.is_normalized:
            self.scale_value = numpy.nanmax(self.data[self.outcome_variable]) - \
                               numpy.nanmin(self.data[self.outcome_variable])

    def normalize_data(self):
        self.data[self.outcome_variable] = self.normalize(self.data[self.outcome_variable].values)

    def normalize(self, values):
        return [(value - self.center_value) / self.scale_value for value in values]

    def denormalize_data(self):
        self.data[self.outcome_variable] = [value * self.scale_value + self.center_value for value in
                                            self.data[self.outcome_variable]]
        self.new_detransformed_data = [value * self.scale_value + self.center_value for value in
                                       self.new_detransformed_data]

    def depower_transform_data(self):
        self.data[self.outcome_variable] = [value ** (1 / self.power_exponent) for value in
                                            self.data[self.outcome_variable]]
        self.new_detransformed_data = [value ** (1 / self.power_exponent) for value in self.new_detransformed_data]

    def dedifference_data_if_needed(self):
        if self.differencing_order > 0:
            self.update_new_correction_values()
            self.data.loc[:, self.outcome_variable] = \
                [self.data.loc[index, self.outcome_variable] + self.differencing_correction_values.loc[index]
                 for index in self.data.index]
            self.new_detransformed_data = [self.new_detransformed_data[i] +
                                           self.new_differencing_correction_values[i] for i in
                                           range(len(self.new_detransformed_data))]

    def update_new_correction_values(self):
        for i in range(len(self.new_detransformed_data)):
            if i + 1 <= self.differencing_order:
                index = self.data.index[-self.differencing_order + i]
                self.new_differencing_correction_values.append(self.data.loc[index, self.outcome_variable] +
                                                               self.differencing_correction_values.loc[index])
            else:
                self.new_differencing_correction_values.append(self.new_transformed_data[i - self.differencing_order] +
                                                               self.new_differencing_correction_values[
                                                                   i - self.differencing_order])
