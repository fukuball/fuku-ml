#encoding=utf8

import FukuML.KernelRidgeRegression as kernel_ridge_regression


class BinaryClassifier(kernel_ridge_regression.BinaryClassifier):

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

        return super(BinaryClassifier, self).score_function(x, W)

    def error_function(self, y_prediction, y_truth):

        return super(BinaryClassifier, self).error_function(y_prediction, y_truth)

    def calculate_avg_error(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        return super(BinaryClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)


class MultiClassifier(kernel_ridge_regression.MultiClassifier):

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

        self.class_list = []
        self.classifier_list = {}
        self.decomposition = 'ovo'

    def load_train_data(self, input_data_file=''):

        return super(MultiClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(MultiClassifier, self).load_test_data(input_data_file)

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        return super(MultiClassifier, self).set_param(lambda_p, gamma, C)

    def init_W(self):

        return super(MultiClassifier, self).init_W()

    def score_function(self, x, W):

        return super(MultiClassifier, self).score_function(x, W)

    def error_function(self, y_prediction, y_truth):

        return super(MultiClassifier, self).error_function(y_prediction, y_truth)

    def calculate_avg_error(self, X, Y, W):

        return super(MultiClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(MultiClassifier, self).calculate_test_data_avg_error()

    def train(self):

        return super(MultiClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(MultiClassifier, self).prediction(input_data, mode)
