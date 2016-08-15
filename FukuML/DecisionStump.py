#encoding=utf8

from __future__ import division
import os
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml


class BinaryClassifier(ml.Learner):

    # too slow for high dimension data, can't do digits multi classifier
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

        self.sign = 1
        self.feature_index = 0
        self.theta = 0
        self.u = None

    def load_train_data(self, input_data_file=''):

        '''
        Load train data
        Please check dataset/pla_binary_train.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/decision_stump_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        '''
        Load test data
        Please check dataset/pocket_pla_binary_test.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/decision_stump_test.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.test_X, self.test_Y

        self.test_X, self.test_Y = utility.DatasetLoader.load(input_data_file)

        if (self.feature_transform_mode == 'polynomial') or (self.feature_transform_mode == 'legendre'):
            self.test_X = self.test_X[:, 1:]

            self.test_X = utility.DatasetLoader.feature_transform(
                self.test_X,
                self.feature_transform_mode,
                self.feature_transform_degree
            )

        return self.test_X, self.test_Y

    def set_param(self, u=None):

        self.u = u

        return self.u

    def init_W(self, mode='normal'):

        '''
        Init the W
        Simple way is init W all zeros
        '''

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.W = np.zeros(self.data_demension)

        if self.u is None:
            self.u = np.array([(1.0 / self.data_num)] * self.data_num)

        return self.W

    def score_function(self, x, W):
        # need refector

        '''
        Score function to calculate score
        '''

        score = self.sign * np.sign(x[self.feature_index] - self.theta)

        return score

    def error_function(self, y_prediction, y_truth):
        # need refector

        '''
        Error function to calculate error
        '''

        if y_prediction != y_truth:
            return 1
        else:
            return 0

    def calculate_avg_error(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        error_in = self.data_num/self.data_num

        for i in range(0, self.train_X.shape[1]):

            dim_X = self.train_X[:, i]
            dim_XY = np.transpose(np.array([dim_X, self.train_Y]))
            sort_index = np.argsort(dim_XY[:, 0])
            sort_dim_XY = dim_XY[sort_index]
            sort_u = self.u[sort_index]

            sort_dim_X = sort_dim_XY[:, 0]
            sort_dim_Y = sort_dim_XY[:, 1]

            thetas = np.array([float("-inf")] + [(sort_dim_X[j] + sort_dim_X[j+1])/2 for j in range(0, self.data_num-1)] + [float("inf")])
            error_in_i = sum(sort_u)
            sign_i = 1
            theta_i = 0.0

            for theta in thetas:
                y_positive = np.where(sort_dim_X > theta, 1, -1)
                y_negative = np.where(sort_dim_X < theta, 1, -1)
                error_positive = sum((y_positive != sort_dim_Y)*sort_u)
                error_negative = sum((y_negative != sort_dim_Y)*sort_u)
                if error_positive > error_negative:
                    if error_in_i > error_negative:
                        error_in_i = error_negative
                        sign_i = -1
                        theta_i = theta
                else:
                    if error_in_i > error_positive:
                        error_in_i = error_positive
                        sign_i = 1
                        theta_i = theta

            if error_in > error_in_i:
                error_in = error_in_i
                self.sign = sign_i
                self.feature_index = i
                self.theta = theta_i

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
