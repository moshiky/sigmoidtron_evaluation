
import time


class Timer(object):

    def __init__(self):
        self.__start_time = None
        self.init()

    def init(self):
        self.__start_time = time.time()

    def get_passed_time(self):
        return time.time() - self.__start_time
