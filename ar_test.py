
import matplotlib.pyplot as plt

from config import Config
from utils.logger import Logger
from utils.dataset_utils import DatasetUtils
from utils.error_metrics import ErrorMetrics
from utils.dict_tools import DictTools
from models.arma import ARMAModel


def main(dataset):
    """
    K - series length
    :param dataset: list of lists. all samples. each element in base list is a list with K float elements
    :return:
    """
    # create logger
    logger = Logger()
    logger.log('AR evaluation begin')

    # iterate all samples
    all_error_metrics = dict()
    for sample in dataset:
        # train AR model
        model = ARMAModel(logger, sample, p=1, q=0, with_c=False)

        # predict all values
        predictions = model.get_fitted_values()

        # # plot predictions vs. sample
        # plt.plot(sample)
        # plt.plot(predictions)
        # plt.show()

        # get error metrics
        error_metrics = ErrorMetrics.get_all_metrics(sample, predictions)
        DictTools.update_dict_with_lists(all_error_metrics, error_metrics)

    # log error metrics' average
    DictTools.log_dict_avg_sorted(logger, all_error_metrics)


def get_dataset():
    # load dataset
    dataset = \
        DatasetUtils.parse_csv(
            Config.DATASET_FILE_PATH,
            should_shuffle=True,
            min_record_length=Config.MIN_RECORD_LENGTH
        )

    # smooth dataset samples
    for sample in dataset:
        DatasetUtils.smooth_series(sample, smoothing_level=1)

    return dataset


if __name__ == '__main__':
    # evaluate dataset
    main(get_dataset())
