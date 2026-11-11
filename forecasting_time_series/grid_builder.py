import itertools

from .processing import PROCESSING_KEYS


class builder:
    """Builds every combination of method parameters × processing settings × training windows.

    Processing settings and training windows are the same for every method, so they are
    combined generically; each method only defines its own parameter grid."""

    def __init__(self, method, method_and_processing_parameter_ranges):
        self.method = method
        self.method_parameter_ranges = method_and_processing_parameter_ranges.get('method_parameter_ranges')
        self.processing_parameter_ranges = method_and_processing_parameter_ranges.get(
            'processing_parameter_ranges')
        # None = expanding window (all data before each test window); a number = fixed length
        self.training_window_range = method_and_processing_parameter_ranges.get('training_window_range', [None])
        self.grid_of_parameter_combinations = []

    # main
    def get_grid_of_parameter_combinations(self):
        """List of {'method_parameters': {...}, 'processing_parameters': {...}, 'training_window': n or None}."""
        self.grid_of_parameter_combinations = [{'method_parameters': method_parameters,
                                                'processing_parameters': processing_parameters,
                                                'training_window': training_window}
                                               for method_parameters in self.get_method_grid()
                                               for processing_parameters in self.get_processing_grid()
                                               for training_window in self.training_window_range]
        return self.grid_of_parameter_combinations

    # subsidiary
    def get_method_grid(self):
        ranges = self.method_parameter_ranges

        if self.method == 'Naive':
            return [{'predictors_length': length,
                     'predictors_offset': offset,
                     'average_type': average_type}
                    for max_length in ranges.get('predictors_maximum_length_range')
                    for length in range(1, max_length + 1)
                    for offset in ranges.get('predictors_offset_range')
                    for average_type in ranges.get('average_type_range')]

        if self.method == 'LEAR':
            # the input set is fixed by design (the Lasso selects among the inputs): only the penalty is searched
            lag_days = ranges.get('lag_days')
            return [{'lag_days': tuple(lag_days) if isinstance(lag_days, (list, tuple)) else lag_days,
                     'lag_resolution': ranges.get('lag_resolution'),
                     'exog_lag_days': {column: tuple(lags) if isinstance(lags, (list, tuple)) else lags
                                       for column, lags in (ranges.get('exog_lag_days') or {}).items()},
                     'exog_lag_transform': ranges.get('exog_lag_transform', 'asinh'),
                     'lag_interactions': tuple(ranges.get('lag_interactions') or ()),
                     'penalty': penalty,
                     'steps_per_day': ranges.get('steps_per_day'),
                     'minimum_calibration_days': ranges.get('minimum_calibration_days')}
                    for penalty in ranges.get('penalty_range')]

        if self.method in ('MLP', 'LSTM'):
            # the inputs are LEAR's (arranged as a sequence of days for the LSTM); the network's settings
            # are searched
            lag_days = ranges.get('lag_days')
            fixed = ('activation', 'batch_size', 'max_epochs', 'patience', 'validation_days', 'refit', 'n_seeds',
                     'seed') + (('lstm_layers', 'sequence_known_inputs', 'sequence_steps', 'head_inputs')
                                if self.method == 'LSTM' else ())
            shared = {'lag_days': tuple(lag_days) if isinstance(lag_days, (list, tuple, range)) else lag_days,
                      'lag_resolution': ranges.get('lag_resolution'),
                      'exog_lag_days': {column: tuple(lags) if isinstance(lags, (list, tuple, range)) else lags
                                        for column, lags in (ranges.get('exog_lag_days') or {}).items()},
                      'exog_lag_transform': ranges.get('exog_lag_transform', 'asinh'),
                      'steps_per_day': ranges.get('steps_per_day'),
                      'minimum_calibration_days': ranges.get('minimum_calibration_days'),
                      **{name: ranges.get(name) for name in fixed}}
            units = ranges.get('lstm_units_range') if self.method == 'LSTM' else [None]
            return [{**shared, **({'lstm_units': n} if n is not None else {}),
                     'hidden_layers': tuple(layers), 'dropout': dropout,
                     'weight_decay': weight_decay, 'learning_rate': learning_rate}
                    for n in units
                    for layers in ranges.get('hidden_layers_range')
                    for dropout in ranges.get('dropout_range')
                    for weight_decay in ranges.get('weight_decay_range')
                    for learning_rate in ranges.get('learning_rate_range')]

        if self.method == 'RF':
            # the MLP's inputs; the forest's settings are searched
            lag_days = ranges.get('lag_days')
            shared = {'lag_days': tuple(lag_days) if isinstance(lag_days, (list, tuple, range)) else lag_days,
                      'lag_resolution': ranges.get('lag_resolution'),
                      'exog_lag_days': {column: tuple(lags) if isinstance(lags, (list, tuple, range)) else lags
                                        for column, lags in (ranges.get('exog_lag_days') or {}).items()},
                      'exog_lag_transform': ranges.get('exog_lag_transform', 'asinh'),
                      'steps_per_day': ranges.get('steps_per_day'),
                      'minimum_calibration_days': ranges.get('minimum_calibration_days'),
                      **{name: ranges.get(name) for name in ('max_samples', 'seed', 'n_jobs')}}
            return [{**shared, 'n_estimators': n_estimators, 'max_depth': max_depth,
                     'min_samples_leaf': min_samples_leaf, 'max_features': max_features,
                     'target_anchor': target_anchor}
                    for target_anchor in ranges.get('target_anchor_range') or [None]
                    for n_estimators in ranges.get('n_estimators_range')
                    for max_depth in ranges.get('max_depth_range')
                    for min_samples_leaf in ranges.get('min_samples_leaf_range')
                    for max_features in ranges.get('max_features_range')]

        raise ValueError(f'no parameter grid defined for method {self.method!r}')

    def get_processing_grid(self):
        ranges = self.processing_parameter_ranges
        return [dict(zip(PROCESSING_KEYS, values))
                for values in itertools.product(*(ranges.get(f'{key}_range') for key in PROCESSING_KEYS))]
