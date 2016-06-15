#encoding=utf8

import FukuML.KernelRidgeRegression as kernel_ridge_regression


class LeastSquaresSVM(kernel_ridge_regression.BinaryClassifier):

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

        self.lambda_p = 0.0001
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.beta = []

    def load_train_data(self, input_data_file=''):

        return super(LeastSquaresSVM, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(LeastSquaresSVM, self).load_test_data(input_data_file)

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        return super(LeastSquaresSVM, self).set_param(lambda_p, gamma, C)

    def init_W(self):

        return super(LeastSquaresSVM, self).init_W()

    def score_function(self, x, W):

        return super(LeastSquaresSVM, self).score_function(x, W)

    def error_function(self, y_prediction, y_truth):

        return super(LeastSquaresSVM, self).error_function(y_prediction, y_truth)

    def calculate_avg_error(self, X, Y, W):

        return super(LeastSquaresSVM, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(LeastSquaresSVM, self).calculate_test_data_avg_error()

    def train(self):

        return super(LeastSquaresSVM, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(LeastSquaresSVM, self).prediction(input_data, mode)
