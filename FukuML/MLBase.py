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

    @abstractmethod
    def __init__(self):
        """init"""
        return

    @abstractmethod
    def load_train_data(self, input_data_file=''):
        """Load train data"""
        return

    def serialize(self):
        """serialize"""
        return pickle.dumps(self, protocol=0)
