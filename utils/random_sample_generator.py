
import numpy as np

from utils.logger import Logger
import sigmoid_generator


def get_random_sample(model_logic, params, length):
    """
    x = 1, 2, ... , length
    :param model_logic:
    :param params:
    :param length:
    :return:
    """
    # initiate series storage
    series = list()

    # calculate noised predictions
    # noise_rate = 0.001
    for x_t in range(1, length + 1):
        # get clean prediction
        prediction = model_logic.predict(params, x_t)

        # # add random noise
        # max_noise_size = noise_rate * prediction
        # noise = 2 * np.random.random() * max_noise_size - max_noise_size
        # prediction += noise

        # store prediction
        series.append(prediction)

    # return series
    return series


def get_random_dataset(model_logic, dataset_size, min_series_length, max_series_length):
    """
    returns dataset with requested number of randomized series.

    :param model_logic:
    :param dataset_size:
    :param min_series_length:
    :param max_series_length:
    :return:
    """
    # initiate dataset
    dataset = list()
    logger = Logger()

    # build one series at a time
    for series_id in range(dataset_size):

        # get random params
        params = model_logic.get_initial_params(first_observation=None)
        # logger.log('ser#{id} : {params}'.format(id=series_id, params=params))

        # randomize length
        series_length = np.random.randint(low=min_series_length, high=max_series_length)
        # sigmoid_generator.plot_sigmoid(model_logic, params, series_length)

        # get series
        series = get_random_sample(model_logic, params, series_length)

        # store sample in dataset
        dataset.append(series)

    # return dataset
    return dataset

