
import numpy as np

from config import Config
from utils.numpy_utils import NumpyUtils


class SigmoidLogic(object):

    """
    Simplest model:
        - hypothesis:
                         b
        s_t = a + ------------------        1 < a < 100
                  c + e ^ (d*t + f)         1 < b < 100
                                            1 < c < 10
                                                d < 0
                                                f > 0

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

        a_param = NumpyUtils.random_number_in_range(1, 10)
        b_param = NumpyUtils.random_number_in_range(1, 10)
        d_param = NumpyUtils.random_number_in_range(-0.8, -0.4)

        if first_observation is None:
            first_value = a_param + 1e-1
            # f_param = NumpyUtils.random_number_in_range(6, 10)
            """
            
            O = a + b / ( c + e ^ (d*x0 + f) )
            
            --> b = ( c + e ^ (d*x0 + f) ) * 
            
            --> b / ( O - a ) = c + e ^ (d*x0 + f)
            
            --> [ b / ( O - a ) ] - c = e ^ (d*x0 + f)
            
            --> ln( [ b / ( O - a ) ] - c ) = d*x0 + f
            
            --> f = ln( [ b / ( O - a ) ] - c ) - d*x0
            
            """
            c_param =\
                min(
                    NumpyUtils.random_number_in_range(0.1, 10),
                    (b_param / (first_value - a_param)) - SigmoidLogic.EPSILON
                )
            f_param = np.log((b_param / (first_value - a_param)) - c_param) - d_param * x0
            f_param = min(f_param, 10)

        else:
            c_param = \
                min(
                    NumpyUtils.random_number_in_range(0.1, 10),
                    (b_param / (first_observation - a_param)) - SigmoidLogic.EPSILON
                )
            f_param = np.log((b_param / (first_observation - a_param)) - c_param) - d_param * x0
            f_param = min(f_param, 10)

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
            # get gradients rounds) + 1))
            grads = self.__get_gradients(params, y_t, x_t)

            # apply gradients to params vector
            params -= (self.__learning_rate / np.sqrt(self.__steps)) * grads
        self.__steps += 1

    @staticmethod
    def __project(params, obs):
        # extract params
        a_param, b_param, c_param, d_param, f_param = params

        # fix params


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



        ################################################

        X = np.float64(a_param)
        Y = np.float64(b_param)
        Z = np.float64(c_param)
        W = np.float64(d_param)
        U = np.float64(f_param)
        P = np.float64(x_t)
        K = np.float64(y_t)

        # X
        dx = 2 * (-K + (Y / (np.exp(P * W + U) + Z)) + X)

        # Y
        dy = (2 * (-K + (Y / (np.exp(P * W + U) + Z)) + X)) / (np.exp(P * W + U) + Z)

        # Z
        dz = - (2 * Y * (-K + (Y / (np.exp(P * W + U) + Z)) + X)) / np.square(np.exp(P * W + U) + Z)

        # W
        dw = - (2 * P * Y * np.exp(P * W + U) * (-K + (Y / (np.exp(P * W + U) + Z)) + X)) / np.square(np.exp(P * W + U) + Z)

        # U
        du = - (2 * Y * np.exp(P * W + U) * (-K + (Y / (np.exp(P * W + U) + Z)) + X)) / np.square(np.exp(P * W + U) + Z)

        ################################################

        # return gradient
        return np.array([
            df_da, df_db, df_dc, df_dd, df_df
        ])
