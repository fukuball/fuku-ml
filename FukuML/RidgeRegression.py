#encoding=utf8

import os
import itertools
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml


class RidgeRegression(ml.Learner):

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

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pocket_pla_binary_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pocket_pla_binary_test.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.test_X, self.test_Y

        self.test_X, self.test_Y = utility.DatasetLoader.load(input_data_file)

        if (self.feature_transform_mode == 'polynomial') or (self.feature_transform_mode == 'legendre'):
            self.test_X = self.test_X[:, 1:]

            self.test_X = utility.DatasetLoader.featureTransform(
                self.test_X,
                self.feature_transform_mode,
                self.feature_transform_degree
            )

        return self.test_X, self.test_Y

    def init_W(self):

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.W = np.zeros(self.data_demension)

        return self.W

    def score_function(self, x, W):

        score = np.inner(x, W)

        return score

    def error_function(self, y_prediction, y_truth):

        error = (y_prediction - y_truth) ** 2

        return error

    def calculate_avg_error(self, X, Y, W):

        return super(RidgeRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(RidgeRegression, self).calculate_avg_error()

    def train(self, lambda_p=0.0001):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        inverse_part = np.linalg.inv(np.dot(self.train_X.transpose(), self.train_X) + lambda_p * np.eye(self.train_X.shape[1]))
        self.W = np.dot(np.dot(inverse_part, self.train_X.transpose()), self.train_Y)

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(RidgeRegression, self).prediction(input_data, mode)


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
