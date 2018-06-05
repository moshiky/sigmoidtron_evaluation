
import numpy as np

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
    EPSILON = 1e-10

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__steps = 1

    @staticmethod
    def get_initial_params(randomize=True):
        if randomize:
            a_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
            b_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, 0.0 - SigmoidLogic.EPSILON)
            c_param = NumpyUtils.random_number_in_range(SigmoidLogic.MIN_RANDOM, SigmoidLogic.MAX_RANDOM)
        else:
            a_param = -1 - np.exp(0.1*10 + 1)
            b_param = -0.1
            c_param = 1.0

        return np.array([a_param, b_param, c_param])

    @staticmethod
    def predict(params, x_t):
        # extract params
        a_param, b_param, c_param = params

        # return prediction
        return 1.0 / (a_param + np.exp(b_param * x_t + c_param))

    def update(self, params, y_t, x_t):
        for i in range(1):
            # get gradients
            grads = self.__get_gradients(params, y_t, x_t)
            print(NumpyUtils.get_vector_size(grads), grads)

            # apply gradients to params vector
            params -= self.__learning_rate * grads
            # self.__steps += 1

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
        a_param, b_param, c_param = params
        exp = np.exp(x_t * b_param + c_param)
        base = 2 * (y_t * exp + y_t * a_param - 1) \
            / np.power(exp + a_param, 3)

        # df/da
        df_da = base

        # df/db
        df_db = x_t * exp * base

        # df/dc
        df_dc = exp * base

        # return gradient
        return np.array([
            df_da, df_db, df_dc
        ])
