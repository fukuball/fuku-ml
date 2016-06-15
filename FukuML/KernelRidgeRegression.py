#encoding=utf8

import numpy as np
import FukuML.Utility as utility
import FukuML.RidgeRegression as ridge_regression


class KernelRidgeRegression(ridge_regression.RidgeRegression):

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

        return super(KernelRidgeRegression, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(KernelRidgeRegression, self).load_test_data(input_data_file)

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        # larger C => weaker regularization, smaller C => stronger regularization
        # smaller lambda => weaker regularization, larger lambda => stronger regularization
        self.lambda_p = lambda_p
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = gamma
        self.Q = 1
        self.C = C

        return self.lambda_p, self.gamma, self.C

    def init_W(self):

        return super(KernelRidgeRegression, self).init_W()

    def score_function(self, x, W):

        x = x[1:]
        original_X = self.train_X[:, 1:]
        score = np.sum(self.beta * utility.Kernel.kernel_matrix_xX(self, x, original_X))

        return score

    def error_function(self, y_prediction, y_truth):

        return super(KernelRidgeRegression, self).error_function(y_prediction, y_truth)

    def calculate_avg_error(self, X, Y, W):

        return super(KernelRidgeRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(KernelRidgeRegression, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        original_X = self.train_X[:, 1:]
        K = utility.Kernel.kernel_matrix(self, original_X)
        I = np.diag(np.ones(self.data_num))

        inverse_part = np.linalg.inv(self.lambda_p * I + K)
        self.beta = np.dot(inverse_part, self.train_Y)

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(KernelRidgeRegression, self).prediction(input_data, mode)


class BinaryClassifier(KernelRidgeRegression):

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

        return super(BinaryClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_test_data(input_data_file)

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        return super(BinaryClassifier, self).set_param(lambda_p, gamma, C)

    def init_W(self):

        return super(BinaryClassifier, self).init_W()

    def score_function(self, x, W):

        score = super(BinaryClassifier, self).score_function(x, W)
        score = np.sign(score)

        return score

    def error_function(self, y_prediction, y_truth):

        if y_prediction != y_truth:
            return 1
        else:
            return 0

    def calculate_avg_error(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        return super(BinaryClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
