
class SigmoidtronModel(object):

    def __init__(self, logger, logic_class):
        self.__logger = logger

        # store logic class
        self.__logic_class = logic_class

        # init params vector
        self.__params = logic_class.get_initial_params()

    def get_prediction(self, x_t):
        # use logic class to generate prediction
        return self.__logic_class.predict(self.__params, x_t)

    def update_params(self, observation, x_t):
        # use logic class to update params
        self.__logic_class.update(self.__params, observation, x_t)
