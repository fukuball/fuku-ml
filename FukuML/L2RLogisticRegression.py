#encoding=utf8

#import operator
#import itertools
import numpy as np
import FukuML.LogisticRegression as logistic_regression


class L2RLogisticRegression(logistic_regression.LogisticRegression):

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

        self.feed_mode = 'batch'
        self.step_eta = 0.126
        self.updates = 2000
        self.lambda_p = 0.0001

    def load_train_data(self, input_data_file=''):

        return super(L2RLogisticRegression, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(L2RLogisticRegression, self).load_test_data(input_data_file)

    def set_param(self, feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001):

        # larger C => weaker regularization, smaller C => stronger regularization
        # smaller lambda => weaker regularization, larger lambda => stronger regularization
        self.feed_mode = feed_mode
        self.step_eta = step_eta
        self.updates = updates
        self.lambda_p = lambda_p

        return self.feed_mode, self.step_eta, self.updates, self.lambda_p

    def init_W(self, mode='normal'):

        return super(L2RLogisticRegression, self).init_W(mode)

    def theta(self, s):

        return super(L2RLogisticRegression, self).theta(s)

    def score_function(self, x, W):

        return super(L2RLogisticRegression, self).score_function(x, W)

    def error_function(self, x, y, W):

        return super(L2RLogisticRegression, self).error_function(x, y, W)

    def calculate_gradient(self, X, Y, W):

        if type(Y) is np.ndarray:
            data_num = len(Y)
        else:
            data_num = 1

        gradient_average = ((2*self.lambda_p)/data_num)*self.W + np.dot(self.theta((-1)*Y*np.dot(W, X.transpose()))*((-1)*Y), X)/data_num

        return gradient_average

    def calculate_avg_error(self, X, Y, W):

        return super(L2RLogisticRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(L2RLogisticRegression, self).calculate_test_data_avg_error()

    def train(self):

        return super(L2RLogisticRegression, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(L2RLogisticRegression, self).prediction(input_data, mode)


class BinaryClassifier(L2RLogisticRegression):

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

        self.feed_mode = 'batch'
        self.step_eta = 0.126
        self.updates = 2000
        self.lambda_p = 0.0001

    def load_train_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_test_data(input_data_file)

    def set_param(self, feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001):

        return super(BinaryClassifier, self).set_param(feed_mode, step_eta, updates)

    def init_W(self, mode='normal'):

        return super(BinaryClassifier, self).init_W(mode)

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

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        return super(BinaryClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
