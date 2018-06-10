
"""
Ariel's params:

    dataset_size = 1000
    for all     -   f(x_t=1) = K
    a ~ U[10, 80]
    b ~ U[20, 90]
    c ~ U[0.1, 0.5]

    900 is train set, 100 is test set

    ANOVA test ... ?
"""

import csv
import time

from models.sigmoid_logics.sig_logic_v4 import SigmoidLogic
from utils import random_sample_generator
from utils.dict_tools import DictTools
from utils.logger import Logger
import ar_eval
import ma_eval
import es_eval
import sig_eval


DATASET_SIZE = 1000
MIN_SERIES_LENGTH = 25
MAX_SERIES_LENGTH = 30


def main():
    # get logger
    logger = Logger()

    # get dataset
    dataset = \
        random_sample_generator.get_random_dataset(SigmoidLogic, DATASET_SIZE, MIN_SERIES_LENGTH, MAX_SERIES_LENGTH)

    # build estimator set
    estimator_set = [
        # ar_eval,
        # ma_eval,
        es_eval,
        sig_eval
    ]

    # run each estimator and get average results
    results = dict()
    for estimator in estimator_set:
        original_results = estimator.run(dataset)
        results[estimator.__name__] = DictTools.get_dict_avg(original_results)
        results[estimator.__name__]['valid_samples'] = len(list(original_results.values())[0])

    # store results to csv
    csv_rows = list()
    sorted_result_keys = sorted(list(results.values())[0].keys())
    csv_rows.append(['estimator name'] + sorted_result_keys)
    for estimator_name in results.keys():
        data_row = [estimator_name] + [results[estimator_name][key_name] for key_name in sorted_result_keys]
        csv_rows.append(data_row)

    output_file_path = r'output/synthetic_data_test_{timestamp}.csv'.format(timestamp=int(time.time()))
    logger.log('output file: {file_path}'.format(file_path=output_file_path))
    with open(output_file_path, 'wt') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(csv_rows)


if __name__ == '__main__':
    main()
