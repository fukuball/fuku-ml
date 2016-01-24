#encoding=utf8

import os
import random
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml


class LogisticRegression(ml.Learner):

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

    def load_train_data(self, input_data_file=''):

        '''
        Load train data
        Please check dataset/logistic_regression_train.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/logistic_regression_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        '''
        Load test data
        Please check dataset/logistic_regression_test.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/logistic_regression_test.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.test_X, self.test_Y

        self.test_X, self.test_Y = utility.DatasetLoader.load(input_data_file)

        return self.test_X, self.test_Y

    def init_W(self):

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

        return self.W

    def theta(self, s):

        '''
        Theta sigmoid function
        '''

        return 1/(1 + np.exp((-1)*s))

    def score_function(self, x, W):
        # need refector

        '''
        Score function to calculate score
        '''

        score = self.theta(np.inner(x, W))

        return score

    def error_function(self, x, y, W):
        # need refector

        '''
        Error function to calculate error: cross entropy error
        '''

        error = np.log(1 + np.exp((-1)*y*np.inner(x, W)))

        return error

    def calculate_gradient(self, X, Y, W):

        if type(Y) is np.ndarray:
            data_num = len(Y)
        else:
            data_num = 1

        gradient_average = np.dot(self.theta((-1)*Y*np.dot(W, X.transpose()))*((-1)*Y), X)/data_num

        return gradient_average

    def calculate_avg_error(self, X, Y, W):

        '''
        Calculate avg error from X, Y, W
        '''

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(X[i], Y[i], W)

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        '''
        Calculate test data avg error
        '''

        avg_error = self.calculate_avg_error(self.test_X, self.test_Y, self.W)

        return avg_error

    def train(self, updates=2000, mode='batch', ita=0.0126):

        '''
        Train Linear Regression Algorithm
        From f(x) = WX
        Find best h(x) = WX similar to f(x)
        Output W

        mode: batch / stochastic
        '''

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        for i in range(0, updates):
            if mode == 'stochastic':
                stochastic_i = random.randint(0, self.data_num-1)
                x = self.train_X[stochastic_i]
                y = self.train_Y[stochastic_i]
                gradient = self.calculate_gradient(x, y, self.W)
            else:
                gradient = self.calculate_gradient(self.train_X, self.train_Y, self.W)
            if np.linalg.norm(gradient) == 0:
                return self.W
            self.W = self.W - ita * gradient

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(LogisticRegression, self).prediction(input_data, mode)


class BinaryClassifier(LogisticRegression):

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

    def load_train_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_test_data(input_data_file)

    def init_W(self):

        return super(BinaryClassifier, self).init_W()

    def theta(self, s):

        return super(BinaryClassifier, self).theta(s)

    def score_function(self, x, W):

        '''
        Score function to calculate score
        '''

        score = super(BinaryClassifier, self).score_function(x, W)
        if score >= 0.5:
            score = 1.0
        else:
            score = -1.0

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

    def calculate_gradient(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_gradient(X, Y, W)

    def calculate_avg_error(self, X, Y, W):

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(self.score_function(X[i], W), Y[i])

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_avg_error()

    def train(self, updates=2000, mode='batch', ita=0.0126):

        return super(BinaryClassifier, self).train(updates, mode, ita)

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
