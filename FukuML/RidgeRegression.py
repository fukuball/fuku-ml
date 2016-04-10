#encoding=utf8

import os
import sys
import itertools
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml


class BinaryClassifier(ml.Learner):

    def __init__(self):

        """init"""

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = []
        self.data_num = 0
        self.data_demension = 0
        self.test_X = []
        self.test_Y = []
        self.feature_transform_mode = ''
        self.feature_transform_degree = 1

    def load_train_data(self, input_data_file=''):
        return

    def load_test_data(self, input_data_file=''):
        return

    def init_W(self):
        return

    def score_function(self, x, W):
        return

    def error_function(self, y_prediction, y_truth):
        return

    def train(self):
        return
