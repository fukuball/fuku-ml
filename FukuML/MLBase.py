#encoding=utf8

import pickle
from abc import ABCMeta, abstractmethod


class Learner(object):
    __metaclass__ = ABCMeta

    status = 'empty'
    train_X = []
    train_Y = []
    W = []
    data_num = 0
    data_demension = 0
    tune_times = 0
    test_X = []
    test_Y = []

    def __init__(self):

        """init"""

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = []
        self.data_num = 0
        self.data_demension = 0
        self.tune_times = 0
        self.test_X = []
        self.test_Y = []

    @abstractmethod
    def load_train_data(self, input_data_file=''):
        """Load train data"""
        return

    def serialize(self):
        """serialize"""
        return pickle.dumps(self, protocol=0)
