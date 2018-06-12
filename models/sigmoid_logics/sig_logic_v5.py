
import numpy as np

from config import Config
from utils.numpy_utils import NumpyUtils


class SigmoidLogic(object):

    """
    Simplest model:
        - hypothesis:
                         b
        s_t = a + ------------------        1 < a < 100
                  c + e ^ (d*(t + f))       1 < b < 100
                                            1 < c < 10
                                                d < 0
                                                f > 0

        a, c E R
        b E R-

        - start params are randomized
        - loss function is MSE
        - with projection after update
    """

    EPSILON = 1e-10
    INFINITY = 1e10

    A_RANGE = [1.0, 3.0]
    B_RANGE = [3.0, 6.0]
    C_RANGE = [EPSILON, 1e8]
    D_RANGE = [EPSILON-1.0, -EPSILON]

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__steps = 1

    @staticmethod
    def get_initial_params(first_observation=None, s_point_x=None):

        # set first value
        if first_observation is not None:
            first_value = np.float64(first_observation)
            x0 = 1
        else:
            x0 = 1
            first_value = NumpyUtils.random_number_in_range(SigmoidLogic.A_RANGE[0], SigmoidLogic.A_RANGE[1])

        # handle a_param
        a_param = first_value - SigmoidLogic.EPSILON

        # randomize b_param
        b_param = NumpyUtils.random_number_in_range(SigmoidLogic.B_RANGE[0], SigmoidLogic.B_RANGE[1])

        # randomize c_param
        c_param = NumpyUtils.random_number_in_range(SigmoidLogic.C_RANGE[0], SigmoidLogic.C_RANGE[1])
        # project c_param
        c_param = min(
            c_param,
            (b_param / (first_value - a_param)) - SigmoidLogic.EPSILON
        )

        # calculate d_param and f_param
        if s_point_x is not None:
            d_param = (np.log(c_param) - np.log((b_param / (first_value - a_param)) - c_param)) / (s_point_x - x0)

        else:
            d_param = NumpyUtils.random_number_in_range(SigmoidLogic.D_RANGE[0], SigmoidLogic.D_RANGE[1])

        # calculate f_param
        f_param = (np.log(
            (b_param / (first_value - a_param)) - c_param
        ) / d_param) - x0

        return np.array([a_param, b_param, c_param, d_param, f_param])

    @staticmethod
    def predict(params, x_t):
        # extract params
        a_param, b_param, c_param, d_param, f_param = params

        # return prediction
        exp = np.exp(d_param * (x_t + f_param))
        return a_param + b_param / (c_param + exp)

    def update(self, params, y_t, x_t):
        rounds = 1
        for i in range(rounds):
            # get gradients rounds) + 1))
            grads = self.__get_gradients(params, y_t, x_t)
            # print('grads', grads)

            # apply gradients to params vector
            params -= grads * self.__learning_rate     # / np.sqrt(self.__steps)
        self.__steps += 1
        # self.__project(params)
        # print('params', params)

    @staticmethod
    def __project(params):
        # extract params
        a_param, b_param, c_param, d_param, f_param = params

        # fix a_param
        params[0] = max(a_param, SigmoidLogic.EPSILON)

        # fix b_param
        params[1] = max(b_param, SigmoidLogic.EPSILON)

        # fix c_param
        params[2] = max(c_param, SigmoidLogic.EPSILON)

        # fix d_param
        params[3] = min(-SigmoidLogic.EPSILON, d_param)

    @staticmethod
    def loss(params, y_t, x_t):
        return np.square(y_t - SigmoidLogic.predict(params, x_t))

    @staticmethod
    def __get_gradients(params, y_t, x_t):
        # extract params
        a_param, b_param, c_param, d_param, f_param = params

        # common parts
        exp = np.exp(d_param * (x_t + f_param))
        exp_p_c = exp + c_param
        base = 2 * (y_t - (b_param / exp_p_c) - a_param)

        # df/da
        df_da = -base

        # df/db
        df_db = -base / exp_p_c

        # df/dc
        df_dc = base * b_param / np.square(exp + c_param)

        # df/dd
        df_dd = base * b_param * (x_t + f_param) * exp / np.square(exp + c_param)

        # df/df
        df_df = base * b_param * d_param * exp / np.square(exp + c_param)

        # return gradient
        return np.array([
            df_da, df_db, df_dc, df_dd, df_df
        ])
