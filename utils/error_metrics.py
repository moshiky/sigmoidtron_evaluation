
from scipy import stats
from sklearn.metrics import mean_squared_error


class ErrorMetrics(object):
    
    @staticmethod
    def get_all_metrics(series_a, series_b):
        r2_value = ErrorMetrics.get_r_squared_value(series_a, series_b)
        mse_value = ErrorMetrics.get_mse_value(series_a, series_b)
        mape_value = ErrorMetrics.get_mre_value(series_a, series_b)

        if r2_value < 0.1 or mse_value > 1000 or mape_value > 100:
            raise Exception('bad prediction')

        return {
            'r2': r2_value,
            'mse': mse_value,
            'mape': mape_value
        }

    @staticmethod
    def get_r_squared_value(series_a, series_b):
        """
        0 <= value <= 1
        higher is better
        """
        # todo: filter all values the same cases
        slope, intercept, r_value, p_value, std_err = stats.linregress(series_a, series_b)
        return r_value ** 2

    @staticmethod
    def get_mse_value(series_a, series_b):
        """
        0 <= value
        lower is better
        """
        # Mean Squared Error (MSE)
        return mean_squared_error(series_a, series_b)

    @staticmethod
    def get_mre_value(series_a, series_b):
        """
        0 <= value
        lower is better
        """
        # Mean Absolute Percentage Error (MAPE)
        errors = \
            [
                abs(float(series_a[i] - series_b[i]) / series_a[i])
                for i in range(len(series_a)) if series_a[i] != 0
            ]
        return float(sum(errors)) / len(errors)
