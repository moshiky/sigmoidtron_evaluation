
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
            a_param = (1/0.1) - np.exp(-0.1*0.1 + 1)
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
        # last_loss = self.loss(params, y_t, x_t)
        # while last_loss > 0.01:
        #     print(last_loss)
            # get gradients
            grads = self.__get_gradients(params, y_t, x_t)
            print(grads)

            # apply gradient clipping
            # grad_size = np.sqrt(np.sum(np.square(grads)))
            # print(grad_size)

            # if grad_size > Config.MAX_GRADIENT_SIZE:
            #     clip_factor = Config.MAX_GRADIENT_SIZE / grad_size
            #     grads *= clip_factor

            # apply gradients to params vector
            params -= (1000/np.sqrt(self.__steps)) * grads
            self.__steps += 1
            # print('y={y_t} : p={pred}'.format(y_t=y_t, pred=self.predict(params, x_t)))

            # # apply projection
            # self.__project(params)
            # last_loss = self.loss(params, y_t, x_t)

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

        # calculate common elements
        exp_part = np.exp(b_param * x_t + c_param)
        bottom_part = np.square(a_param + exp_part)
        loss_base = y_t - (1 / (a_param + exp_part))

        # df/da
        df_da = (2 * loss_base) / bottom_part

        # df/db
        df_db = exp_part * x_t * df_da

        # df/dc
        df_dc = exp_part * df_da

        # return gradient
        return np.array([
            df_da, df_db, df_dc
        ])
