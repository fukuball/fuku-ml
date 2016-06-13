#encoding=utf8

import numpy as np
import FukuML.L2RLogisticRegression as l2r_logistic_regression


class KernelLogisticRegression(l2r_logistic_regression.L2RLogisticRegression):

    def __init__(self):

        return super(KernelLogisticRegression, self).__init__()

    def load_train_data(self, input_data_file=''):

        return super(KernelLogisticRegression, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(KernelLogisticRegression, self).load_test_data(input_data_file)

    def set_param(self, feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001):

        return super(KernelLogisticRegression, self).set_param(feed_mode, step_eta, updates, lambda_p)

    def init_W(self, mode='normal'):

        return super(KernelLogisticRegression, self).init_W(mode)

    def theta(self, s):

        return super(KernelLogisticRegression, self).theta(s)

    def score_function(self, x, W):

        return super(KernelLogisticRegression, self).score_function(x, W)

    def error_function(self, x, y, W):

        return super(KernelLogisticRegression, self).error_function(x, y, W)

    def calculate_gradient(self, X, Y, W):

        return super(KernelLogisticRegression, self).calculate_gradient(X, Y, W)

    def calculate_avg_error(self, X, Y, W):

        return super(KernelLogisticRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(KernelLogisticRegression, self).calculate_test_data_avg_error()

    def train(self):

        return super(KernelLogisticRegression, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(KernelLogisticRegression, self).prediction(input_data, mode)
