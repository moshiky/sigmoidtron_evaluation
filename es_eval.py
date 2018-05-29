
import matplotlib.pyplot as plt

from config import Config
from utils.logger import Logger
from utils.dataset_utils import DatasetUtils
from utils.error_metrics import ErrorMetrics
from utils.dict_tools import DictTools
from utils.timer import Timer
from models.exponential_smoothing import ExponentialSmoothingModel


def main(dataset):
    """
    K - series length
    :param dataset: list of lists. all samples. each element in base list is a list with K float elements
    :return:
    """
    # create logger
    logger = Logger()
    logger.log('ES evaluation begin')

    # iterate all samples
    all_error_metrics = dict()
    timer = Timer()
    for sample_idx, sample in enumerate(dataset):
        # log progress
        if sample_idx % Config.SAMPLE_RECORD_INTERVAL == 0:
            logger.log('sample #{sample_idx}'.format(sample_idx=sample_idx))

        try:
            # train AR model
            model = ExponentialSmoothingModel(logger, sample, p=1, alpha=0.6)

            # predict all values
            predictions = model.get_fitted_values()

            # # plot predictions vs. sample
            # plt.plot(sample)
            # plt.plot(predictions)
            # plt.show()

            # get error metrics
            error_metrics = ErrorMetrics.get_all_metrics(sample, predictions)
            DictTools.update_dict_with_lists(all_error_metrics, error_metrics)

        except Exception as ex:
            logger.log('failed to fit sample #{sample_idx}. error: {ex}'.format(sample_idx=sample_idx, ex=ex))

    # log error metrics' average
    logger.log('total time: {time_passed}'.format(time_passed=timer.get_passed_time()))
    DictTools.log_dict_avg_sorted(logger, all_error_metrics)


def get_dataset():
    # load dataset
    dataset = \
        DatasetUtils.parse_csv(
            Config.DATASET_FILE_PATH,
            should_shuffle=False,
            min_record_length=Config.MIN_RECORD_LENGTH
        )

    # smooth dataset samples
    for sample in dataset:
        DatasetUtils.smooth_series(sample, smoothing_level=1)

    return dataset


if __name__ == '__main__':
    # evaluate dataset
    main(get_dataset())
