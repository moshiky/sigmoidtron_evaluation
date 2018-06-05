
import numpy as np


class NumpyUtils(object):

    @staticmethod
    def random_number_in_range(range_min, range_max):
        return range_min + np.random.random() * (range_max - range_min)

    @staticmethod
    def get_vector_size(vec):
        return np.sqrt(np.sum(np.square(vec)))
