
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


class ExponentialSmoothingModel(object):

    def __init__(self, logger, train_set, p, alpha):
        self.__logger = logger

        # set AR component order
        self.__p = p

        # set alpha
        self.__alpha = alpha

        # init train data members
        self.__train_set = train_set
        self.__model = None

        # learn model params
        self.__learn_model_params()

    def __learn_model_params(self, print_settings=False, start_params=None):
        self.__model = \
            SimpleExpSmoothing(np.asarray(self.__train_set)).fit(smoothing_level=self.__alpha, optimized=False)

    def get_fitted_values(self):
        return self.__model.fittedvalues
