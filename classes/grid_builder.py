class builder:

    def __init__(self, method, method_and_processing_parameter_ranges):
        self.method = method
        self.method_and_processing_parameter_ranges = method_and_processing_parameter_ranges
        self.method_parameter_ranges = self.method_and_processing_parameter_ranges.get('method_parameter_ranges')
        self.processing_parameter_ranges = self.method_and_processing_parameter_ranges.get(
            'processing_parameter_ranges')

        self.differencing_order_range = self.processing_parameter_ranges.get('differencing_order_range')
        self.power_exponent_range = self.processing_parameter_ranges.get('power_exponent_range')
        self.is_normalized_range = self.processing_parameter_ranges.get('is_normalized_range')

        self.grid_of_parameter_combinations = []

    # main
    def get_grid_of_parameter_combinations(self):
        parameter_combination = {}

        if self.method == 'Naive':
            self.predictors_maximum_length_range = self.method_parameter_ranges.get('predictors_maximum_length_range')
            self.predictors_offset_range = self.method_parameter_ranges.get('predictors_offset_range')
            self.average_type_range = self.method_parameter_ranges.get('average_type_range')

            for max_length in self.predictors_maximum_length_range:
                for length in range(1, max_length + 1):
                    for offset in self.predictors_offset_range:
                        for average_type in self.average_type_range:
                            for differencing_order in self.differencing_order_range:
                                for power_exponent in self.power_exponent_range:
                                    for is_normalized in self.is_normalized_range:
                                        parameter_combination['predictors_length'] = length
                                        parameter_combination['predictors_offset'] = offset
                                        parameter_combination['average_type'] = average_type
                                        parameter_combination['differencing_order'] = differencing_order
                                        parameter_combination['power_exponent'] = power_exponent
                                        parameter_combination['is_normalized'] = is_normalized
                                        self.grid_of_parameter_combinations.append(parameter_combination.copy())

        if self.method == 'ETS':
            self.trend_component_range = self.method_parameter_ranges.get('trend_component_range')
            self.is_trend_damped_range = self.method_parameter_ranges.get('is_trend_damped_range')
            self.seasonal_component_range = self.method_parameter_ranges.get('seasonal_component_range')
            self.seasonal_period_range = self.method_parameter_ranges.get('seasonal_period_range')

            for trend_component in self.trend_component_range:
                for is_trend_damped in self.is_trend_damped_range:
                    for seasonal_component in self.seasonal_component_range:
                        for seasonal_period in self.seasonal_period_range:
                            for differencing_order in self.differencing_order_range:
                                for power_exponent in self.power_exponent_range:
                                    for is_normalized in self.is_normalized_range:
                                        if (seasonal_component is not None and seasonal_period > 1) or (
                                                seasonal_component is None):
                                            if (trend_component is not None and
                                                trend_component == seasonal_component) or \
                                                    (trend_component is None and not is_trend_damped):
                                                parameter_combination['trend_component'] = trend_component
                                                parameter_combination['is_trend_damped'] = is_trend_damped
                                                parameter_combination['seasonal_component'] = seasonal_component
                                                parameter_combination['seasonal_period'] = seasonal_period
                                                parameter_combination['differencing_order'] = differencing_order
                                                parameter_combination['power_exponent'] = power_exponent
                                                parameter_combination['is_normalized'] = is_normalized
                                                self.grid_of_parameter_combinations.append(parameter_combination.copy())

        if self.method == 'ARIMA':
            self.ar_terms_range = self.method_parameter_ranges.get('ar_terms_range')
            self.seasonal_ar_terms_range = self.method_parameter_ranges.get('seasonal_ar_terms_range')
            self.differencing_terms_range = self.method_parameter_ranges.get('differencing_terms_range')
            self.seasonal_differencing_terms_range = self.method_parameter_ranges.get(
                'seasonal_differencing_terms_range')
            self.ma_terms_range = self.method_parameter_ranges.get('ma_terms_range')
            self.seasonal_ma_terms_range = self.method_parameter_ranges.get('seasonal_ma_terms_range')
            self.seasonal_period_range = self.method_parameter_ranges.get('seasonal_period_range')
            self.trend_range = self.method_parameter_ranges.get('trend_range')

            for ar_terms in self.ar_terms_range:
                for seasonal_ar_terms in self.seasonal_ar_terms_range:
                    for differencing_terms in self.differencing_terms_range:
                        for seasonal_differencing_terms in self.seasonal_differencing_terms_range:
                            for ma_terms in self.ma_terms_range:
                                for seasonal_ma_terms in self.seasonal_ma_terms_range:
                                    for seasonal_period in self.seasonal_period_range:
                                        for trend in self.trend_range:
                                            for differencing_order in [0]:
                                                for power_exponent in self.power_exponent_range:
                                                    for is_normalized in self.is_normalized_range:
                                                        parameter_combination['ar_terms'] = ar_terms
                                                        parameter_combination['differencing_terms'] = differencing_terms
                                                        parameter_combination['ma_terms'] = ma_terms
                                                        parameter_combination['trend'] = trend
                                                        if seasonal_period == 0 or \
                                                                (seasonal_ar_terms == 0 and
                                                                 seasonal_differencing_terms == 0 and
                                                                 seasonal_ma_terms == 0):
                                                            parameter_combination['seasonal_ar_terms'] = 0
                                                            parameter_combination['seasonal_differencing_terms'] = 0
                                                            parameter_combination['seasonal_ma_terms'] = 0
                                                            parameter_combination['seasonal_period'] = 0
                                                        else:
                                                            parameter_combination['seasonal_ar_terms'] = \
                                                                seasonal_ar_terms
                                                            parameter_combination['seasonal_differencing_terms'] = \
                                                                seasonal_differencing_terms
                                                            parameter_combination['seasonal_ma_terms'] = \
                                                                seasonal_ma_terms
                                                            parameter_combination['seasonal_period'] = seasonal_period
                                                        parameter_combination['differencing_order'] = differencing_order
                                                        parameter_combination['power_exponent'] = power_exponent
                                                        parameter_combination['is_normalized'] = is_normalized
                                                        self.grid_of_parameter_combinations.append(
                                                            parameter_combination.copy()
                                                        )

        if self.method == 'RF':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_output_leads_range = self.method_parameter_ranges.get('number_output_leads_range')
            self.minimum_samples_leaf_range = self.method_parameter_ranges.get('minimum_samples_leaf_range')
            self.maximum_number_features_range = self.method_parameter_ranges.get('maximum_number_features_range')
            self.number_trees_range = self.method_parameter_ranges.get('number_trees_range')

            for number_input_lags in self.number_input_lags_range:
                for number_output_leads in self.number_output_leads_range:
                    for minimum_samples_leaf in self.minimum_samples_leaf_range:
                        for maximum_number_features in self.maximum_number_features_range:
                            for number_trees in self.number_trees_range:
                                for differencing_order in self.differencing_order_range:
                                    for power_exponent in self.power_exponent_range:
                                        for is_normalized in self.is_normalized_range:
                                            parameter_combination['number_input_lags'] = number_input_lags
                                            parameter_combination['number_output_leads'] = number_output_leads
                                            parameter_combination['minimum_samples_leaf'] = minimum_samples_leaf
                                            parameter_combination['maximum_number_features'] = maximum_number_features
                                            parameter_combination['number_trees'] = number_trees
                                            parameter_combination['differencing_order'] = differencing_order
                                            parameter_combination['power_exponent'] = power_exponent
                                            parameter_combination['is_normalized'] = is_normalized
                                            self.grid_of_parameter_combinations.append(parameter_combination.copy())

        if self.method == 'MLP':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_hidden_nodes_range = self.method_parameter_ranges.get('number_hidden_nodes_range')
            self.number_epochs_range = self.method_parameter_ranges.get('number_epochs_range')
            self.batch_size_range = self.method_parameter_ranges.get('batch_size_range')

            for number_input_lags in self.number_input_lags_range:
                for number_hidden_nodes in self.number_hidden_nodes_range:
                    for number_epochs in self.number_epochs_range:
                        for batch_size in self.batch_size_range:
                            for differencing_order in self.differencing_order_range:
                                for power_exponent in self.power_exponent_range:
                                    for is_normalized in self.is_normalized_range:
                                        parameter_combination['number_input_lags'] = number_input_lags
                                        parameter_combination['number_hidden_nodes'] = number_hidden_nodes
                                        parameter_combination['number_epochs'] = number_epochs
                                        parameter_combination['batch_size'] = batch_size
                                        parameter_combination['differencing_order'] = differencing_order
                                        parameter_combination['power_exponent'] = power_exponent
                                        parameter_combination['is_normalized'] = is_normalized
                                        self.grid_of_parameter_combinations.append(parameter_combination.copy())

        if self.method == 'CNN':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_filters_range = self.method_parameter_ranges.get('number_filters_range')
            self.kernel_size_range = self.method_parameter_ranges.get('kernel_size_range')
            self.number_epochs_range = self.method_parameter_ranges.get('number_epochs_range')
            self.batch_size_range = self.method_parameter_ranges.get('batch_size_range')

            for number_input_lags in self.number_input_lags_range:
                for number_filters in self.number_filters_range:
                    for kernel_size in self.kernel_size_range:
                        for number_epochs in self.number_epochs_range:
                            for batch_size in self.batch_size_range:
                                for differencing_order in self.differencing_order_range:
                                    for power_exponent in self.power_exponent_range:
                                        for is_normalized in self.is_normalized_range:
                                            parameter_combination['number_input_lags'] = number_input_lags
                                            parameter_combination['number_filters'] = number_filters
                                            parameter_combination['kernel_size'] = kernel_size
                                            parameter_combination['number_epochs'] = number_epochs
                                            parameter_combination['batch_size'] = batch_size
                                            parameter_combination['differencing_order'] = differencing_order
                                            parameter_combination['power_exponent'] = power_exponent
                                            parameter_combination['is_normalized'] = is_normalized
                                            self.grid_of_parameter_combinations.append(parameter_combination.copy())

        if self.method == 'LSTM':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_lstm_units_range = self.method_parameter_ranges.get('number_lstm_units_range')
            self.number_epochs_range = self.method_parameter_ranges.get('number_epochs_range')
            self.batch_size_range = self.method_parameter_ranges.get('batch_size_range')

            for number_input_lags in self.number_input_lags_range:
                for number_lstm_units in self.number_lstm_units_range:
                    for number_epochs in self.number_epochs_range:
                        for batch_size in self.batch_size_range:
                            for differencing_order in self.differencing_order_range:
                                for power_exponent in self.power_exponent_range:
                                    for is_normalized in self.is_normalized_range:
                                        parameter_combination['number_input_lags'] = number_input_lags
                                        parameter_combination['number_lstm_units'] = number_lstm_units
                                        parameter_combination['number_epochs'] = number_epochs
                                        parameter_combination['batch_size'] = batch_size
                                        parameter_combination['differencing_order'] = differencing_order
                                        parameter_combination['power_exponent'] = power_exponent
                                        parameter_combination['is_normalized'] = is_normalized
                                        self.grid_of_parameter_combinations.append(parameter_combination.copy())

        if self.method == 'CNN-LSTM':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_subsequences_range = self.method_parameter_ranges.get('number_subsequences_range')
            self.number_filters_range = self.method_parameter_ranges.get('number_filters_range')
            self.kernel_size_range = self.method_parameter_ranges.get('kernel_size_range')
            self.number_lstm_units_range = self.method_parameter_ranges.get('number_lstm_units_range')
            self.number_epochs_range = self.method_parameter_ranges.get('number_epochs_range')
            self.batch_size_range = self.method_parameter_ranges.get('batch_size_range')

            for number_input_lags in self.number_input_lags_range:
                for number_subsequences in self.number_subsequences_range:
                    if number_input_lags % number_subsequences == 0:
                        for number_filters in self.number_filters_range:
                            for kernel_size in self.kernel_size_range:
                                for number_lstm_units in self.number_lstm_units_range:
                                    for number_epochs in self.number_epochs_range:
                                        for batch_size in self.batch_size_range:
                                            for differencing_order in self.differencing_order_range:
                                                for power_exponent in self.power_exponent_range:
                                                    for is_normalized in self.is_normalized_range:
                                                        parameter_combination['number_input_lags'] = number_input_lags
                                                        parameter_combination['number_subsequences'] = number_subsequences
                                                        parameter_combination['number_filters'] = number_filters
                                                        parameter_combination['kernel_size'] = kernel_size
                                                        parameter_combination['number_lstm_units'] = number_lstm_units
                                                        parameter_combination['number_epochs'] = number_epochs
                                                        parameter_combination['batch_size'] = batch_size
                                                        parameter_combination['differencing_order'] = differencing_order
                                                        parameter_combination['power_exponent'] = power_exponent
                                                        parameter_combination['is_normalized'] = is_normalized
                                                        self.grid_of_parameter_combinations.append(
                                                            parameter_combination.copy())

        if self.method == 'ConvLSTM':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_subsequences_range = self.method_parameter_ranges.get('number_subsequences_range')
            self.number_filters_range = self.method_parameter_ranges.get('number_filters_range')
            self.kernel_size_range = self.method_parameter_ranges.get('kernel_size_range')
            self.number_epochs_range = self.method_parameter_ranges.get('number_epochs_range')
            self.batch_size_range = self.method_parameter_ranges.get('batch_size_range')

            for number_input_lags in self.number_input_lags_range:
                for number_subsequences in self.number_subsequences_range:
                    if number_input_lags % number_subsequences == 0:
                        for number_filters in self.number_filters_range:
                            for kernel_size in self.kernel_size_range:
                                for number_epochs in self.number_epochs_range:
                                    for batch_size in self.batch_size_range:
                                        for differencing_order in self.differencing_order_range:
                                            for power_exponent in self.power_exponent_range:
                                                for is_normalized in self.is_normalized_range:
                                                    parameter_combination['number_input_lags'] = number_input_lags
                                                    parameter_combination['number_subsequences'] = number_subsequences
                                                    parameter_combination['number_filters'] = number_filters
                                                    parameter_combination['kernel_size'] = kernel_size
                                                    parameter_combination['number_epochs'] = number_epochs
                                                    parameter_combination['batch_size'] = batch_size
                                                    parameter_combination['differencing_order'] = differencing_order
                                                    parameter_combination['power_exponent'] = power_exponent
                                                    parameter_combination['is_normalized'] = is_normalized
                                                    self.grid_of_parameter_combinations.append(
                                                        parameter_combination.copy())

        if self.method == 'twoDLSTM':
            self.number_input_lags_range = self.method_parameter_ranges.get('number_input_lags_range')
            self.number_lstm_units_layer_1_range = self.method_parameter_ranges.get(
                'number_lstm_units_layer_1_range')
            self.number_lstm_units_layer_2_range = self.method_parameter_ranges.get(
                'number_lstm_units_layer_2_range')
            self.number_epochs_range = self.method_parameter_ranges.get('number_epochs_range')
            self.batch_size_range = self.method_parameter_ranges.get('batch_size_range')

            for number_input_lags in self.number_input_lags_range:
                for number_lstm_units_layer_1 in self.number_lstm_units_layer_1_range:
                    for number_lstm_units_layer_2 in self.number_lstm_units_layer_2_range:
                        for number_epochs in self.number_epochs_range:
                            for batch_size in self.batch_size_range:
                                for differencing_order in self.differencing_order_range:
                                    for power_exponent in self.power_exponent_range:
                                        for is_normalized in self.is_normalized_range:
                                            parameter_combination['number_input_lags'] = number_input_lags
                                            parameter_combination['number_lstm_units_layer_1'] = \
                                                number_lstm_units_layer_1
                                            parameter_combination['number_lstm_units_layer_2'] = \
                                                number_lstm_units_layer_2
                                            parameter_combination['number_epochs'] = number_epochs
                                            parameter_combination['batch_size'] = batch_size
                                            parameter_combination['differencing_order'] = differencing_order
                                            parameter_combination['power_exponent'] = power_exponent
                                            parameter_combination['is_normalized'] = is_normalized
                                            self.grid_of_parameter_combinations.append(parameter_combination.copy())

        return self.grid_of_parameter_combinations
