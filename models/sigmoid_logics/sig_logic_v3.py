
import numpy as np

from config import Config
from utils.numpy_utils import NumpyUtils


class SigmoidLogic(object):

    """
    Simplest model:
        - hypothesis:
                      1
        s_t = ------------------
               a + e ^ (b*t + c)

        a, c E R
        b E R-

        - start params are randomized
        - loss function is MSE
        - with projection after update
    """

    MIN_RANDOM = -0.1
    MAX_RANDOM = 0.1
    EPSILON = 1e-20

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__steps = 1

    @staticmethod
    def get_initial_params(first_observation=None, x0=1):
        if first_observation is None:
            a_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
            b_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
            c_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
            d_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
            f_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
        else:
            a_param = -0.5
            b_param = 0.9
            d_param = -0.2
            f_param = 0.4

            c_param = (b_param / (first_observation - a_param)) - np.exp(d_param * x0 + f_param)

        return np.array([a_param, b_param, c_param, d_param, f_param])

    @staticmethod
    def predict(params, x_t):
        # extract params
        a_param, b_param, c_param, d_param, f_param = params

        # return prediction
        exp = min(max(np.exp(d_param * x_t + f_param), SigmoidLogic.EPSILON), 1e20)
        return a_param + b_param / (c_param + exp)

    def update(self, params, y_t, x_t):
        rounds = 10
        for i in range(rounds):
            # get gradients
            y_factor = 1 + (0.2 / (int(self.__steps / rounds) + 1))
            grads = self.__get_gradients(params, y_t * y_factor, x_t)

            # grad_size = NumpyUtils.get_vector_size(grads)
            # if grad_size > Config.MAX_GRADIENT_SIZE:
            #     clip_factor = Config.MAX_GRADIENT_SIZE / grad_size
            #     grads *= clip_factor

            # apply gradients to params vector
            params -= (self.__learning_rate / np.sqrt(self.__steps)) * grads
            self.__steps += 1

    @staticmethod
    def __project(params):
        # fix b_param
        params[1] = np.min([params[1], 0.0 - SigmoidLogic.EPSILON])

    @staticmethod
    def loss(params, y_t, x_t):
        return np.square(y_t - SigmoidLogic.predict(params, x_t))

    @staticmethod
    def __get_gradients(params, y_t, x_t):
        # extract params
        a_param, b_param, c_param, d_param, f_param = params

        # common parts
        exp = min(max(np.exp(d_param * x_t + f_param), SigmoidLogic.EPSILON), 1e20)
        base = 2 * (-y_t + (b_param / (exp + c_param)) + a_param)

        # df/da
        df_da = base

        # df/db
        df_db = base / (exp + c_param)

        # df/dc
        df_dc = (-1) * base * b_param / np.square(exp + c_param)

        # df/dd
        df_dd = (-1) * base * x_t * b_param * exp / np.square(exp + c_param)

        # df/df
        df_df = (-1) * base * b_param * exp / np.square(exp + c_param)

        # return gradient
        return np.array([
            df_da, df_db, df_dc, df_dd, df_df
        ])
