
import matplotlib.pyplot as plt

from config import Config
from utils.logger import Logger
from utils.dataset_utils import DatasetUtils
from utils.error_metrics import ErrorMetrics
from utils.dict_tools import DictTools
from utils.timer import Timer
from models.sigmoidtron import SigmoidtronModel
from models.sigmoid_logics.sig_logic_v5 import SigmoidLogic


def run(dataset):
    """
    K - series length
    :param dataset: list of lists. all samples. each element in base list is a list with K float elements
    :return:
    """
    # create logger
    logger = Logger()
    logger.log('Sigmoidtron evaluation begin')

    # iterate all samples
    all_error_metrics = dict()
    timer = Timer()
    for sample_idx, sample in enumerate(dataset):
        # log progress
        if sample_idx % Config.SAMPLE_RECORD_INTERVAL == 0:
            logger.log('sample #{sample_idx}'.format(sample_idx=sample_idx))

        try:
            sample_data, original_params = sample

            # train AR model
            model = SigmoidtronModel(logger, sample_data[0], SigmoidLogic(Config.LEARNING_RATE))

            # predict all values
            predictions = list()
            print('############## new series, real params: ', original_params)
            for in_sample_idx in range(1, len(sample_data)):
                x_t = in_sample_idx + 1
                model_prediction = model.get_prediction(x_t)
                # print(x_t, '-->', model_prediction)

                predictions.append(model_prediction)
                model.update_params(sample_data[in_sample_idx], x_t)

            # plot predictions vs. sample
            plt.plot(range(2, len(predictions)+2), sample_data[1:])
            plt.plot(range(2, len(predictions)+2), predictions)
            plt.show()

            # get error metrics
            error_metrics = ErrorMetrics.get_all_metrics(sample_data[1:], predictions)
            DictTools.update_dict_with_lists(all_error_metrics, error_metrics)

        except Exception as ex:
            logger.log('failed to fit sample #{sample_idx}. error: {ex}'.format(sample_idx=sample_idx, ex=ex))
            # raise ex

    # log error metrics' average
    logger.log('total time: {time_passed}'.format(time_passed=timer.get_passed_time()))
    DictTools.log_dict_avg_sorted(logger, all_error_metrics)

    # return error metrics
    return all_error_metrics


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
    run(get_dataset())
