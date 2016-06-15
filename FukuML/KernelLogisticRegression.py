#encoding=utf8

import random
import numpy as np
import FukuML.Utility as utility
import FukuML.L2RLogisticRegression as l2r_logistic_regression


class KernelLogisticRegression(l2r_logistic_regression.L2RLogisticRegression):

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
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.beta = []

    def load_train_data(self, input_data_file=''):

        return super(KernelLogisticRegression, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(KernelLogisticRegression, self).load_test_data(input_data_file)

    def set_param(self, feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001, gamma=1, C=0.1):

        # larger C => weaker regularization, smaller C => stronger regularization
        # smaller lambda => weaker regularization, larger lambda => stronger regularization
        self.feed_mode = feed_mode
        self.step_eta = step_eta
        self.updates = updates
        self.lambda_p = lambda_p
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = gamma
        self.Q = 1
        self.C = C

        return self.feed_mode, self.step_eta, self.updates, self.lambda_p, self.gamma, self.C

    def init_W(self, mode='normal'):

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.W = np.zeros(self.data_demension)
        self.beta = np.zeros(self.data_num)

        return self.W

    def theta(self, s):

        return super(KernelLogisticRegression, self).theta(s)

    def score_function(self, x, W):

        x = x[1:]
        original_X = self.train_X[:, 1:]
        '''
        score = 0
        for i in range(len(self.beta)):
            if (self.svm_kernel == 'polynomial_kernel' or self.svm_kernel == 'soft_polynomial_kernel'):
                score += self.beta[i] * utility.Kernel.polynomial_kernel(self, original_X[i], x)
            elif (self.svm_kernel == 'gaussian_kernel' or self.svm_kernel == 'soft_gaussian_kernel'):
                score += self.beta[i] * utility.Kernel.gaussian_kernel(self, original_X[i], x)
        score = self.theta(score)
        '''
        score = np.sum(self.beta * utility.Kernel.kernel_matrix_xX(self, x, original_X))
        score = self.theta(score)

        return score

    def error_function(self, x, y, W):

        x = x[1:]
        original_X = self.train_X[:, 1:]
        score = np.sum(self.beta * utility.Kernel.kernel_matrix_xX(self, x, original_X))
        error = np.log(1 + np.exp((-1)*y*score))

        return error

    def calculate_gradient(self, X, Y, beta):

        if type(Y) is np.ndarray:
            data_num = len(Y)
            original_X = X[:, 1:]
            K = utility.Kernel.kernel_matrix(self, original_X)
        else:
            data_num = 1
            original_x = X[1:]
            original_X = self.train_X[:, 1:]
            K = utility.Kernel.kernel_matrix_xX(self, original_x, original_X)

        gradient_average = ((2*self.lambda_p)/data_num)*np.dot(beta, K) + np.dot(self.theta((-1)*Y*np.dot(beta, K))*((-1)*Y), K)/data_num

        print('calculate gradient descent...')

        return gradient_average

    def calculate_avg_error(self, X, Y, W):

        return super(KernelLogisticRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(KernelLogisticRegression, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        for i in range(0, self.updates):
            if self.feed_mode == 'stochastic':
                stochastic_i = random.randint(0, self.data_num-1)
                x = self.train_X[stochastic_i]
                y = self.train_Y[stochastic_i]
                gradient = self.calculate_gradient(x, y, self.beta)
            else:
                gradient = self.calculate_gradient(self.train_X, self.train_Y, self.beta)
            if np.linalg.norm(gradient) == 0:
                return self.W
            self.beta = self.beta - self.step_eta * gradient

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(KernelLogisticRegression, self).prediction(input_data, mode)
