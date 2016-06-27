#encoding=utf8

import os
import numpy as np
import cvxopt
import cvxopt.solvers
import FukuML.Utility as utility
import FukuML.MLBase as ml


class SupportVectorRegression(ml.Learner):

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

        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 1
        self.epsilon = 0.1
        self.alpha_upper = []
        self.alpha_lower = []
        self.beta = []
        self.sv_index = []
        self.sv_beta = []
        self.sv_X = []
        self.sv_Y = []
        self.free_sv_index = []
        self.free_sv_beta = []
        self.free_sv_X = []
        self.free_sv_Y = []
        self.sv_avg_b = 0

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

            self.test_X = utility.DatasetLoader.feature_transform(
                self.test_X,
                self.feature_transform_mode,
                self.feature_transform_degree
            )

        return self.test_X, self.test_Y

    def set_param(self, svm_kernel='soft_gaussian_kernel', zeta=0, gamma=1, Q=1, C=1, epsilon=0.1):

        # larger C => weaker regularization, smaller C => stronger regularization
        self.svm_kernel = svm_kernel
        self.zeta = zeta
        self.gamma = gamma
        self.Q = Q
        self.C = C
        self.epsilon = epsilon

        return self.svm_kernel

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

        x = x[1:]
        score = np.sum(self.sv_beta * utility.Kernel.kernel_matrix_xX(self, x, self.sv_X))

        return score

    def error_function(self, y_prediction, y_truth):

        error = (y_prediction - y_truth) ** 2

        return error

    def calculate_avg_error(self, X, Y, W):

        return super(SupportVectorRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(SupportVectorRegression, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        original_X = self.train_X[:, 1:]

        K = utility.Kernel.kernel_matrix(self, original_X)

        # P = Q, q = p, G = -A, h = -c

        P = cvxopt.matrix(np.bmat([[K, -K], [-K, K]]))
        q = cvxopt.matrix(np.bmat([self.epsilon-self.train_Y, self.epsilon+self.train_Y]).reshape((-1, 1)))
        G = cvxopt.matrix(np.bmat([[-np.eye(2*self.data_num)], [np.eye(2*self.data_num)]]))
        h = cvxopt.matrix(np.bmat([[np.zeros((2*self.data_num, 1))], [self.C*np.ones((2*self.data_num, 1))]]))
        A = cvxopt.matrix(np.append(np.ones(self.data_num), -1 * np.ones(self.data_num)), (1, 2*self.data_num))
        b = cvxopt.matrix(0.0)
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), A, b)

        # Lagrange multipliers
        alpha = np.array(solution['x']).reshape((2, -1))
        self.alpha_upper = alpha[0]
        self.alpha_lower = alpha[1]
        self.beta = self.alpha_upper - self.alpha_lower

        sv = abs(self.beta) > 1e-5
        self.sv_index = np.arange(len(self.beta))[sv]
        self.sv_beta = self.beta[sv]
        self.sv_X = original_X[sv]
        self.sv_Y = self.train_Y[sv]

        free_sv = np.logical_and(self.beta > 1e-5, self.beta < self.C)
        self.free_sv_index = np.arange(len(self.beta))[free_sv]
        self.free_sv_beta = self.beta[free_sv]
        self.free_sv_X = original_X[free_sv]
        self.free_sv_Y = self.train_Y[free_sv]

        short_b = (np.sum(self.free_sv_Y) - np.sum(np.ravel(self.free_sv_beta * utility.Kernel.kernel_matrix(self, self.free_sv_X)))) / len(self.free_sv_beta)

        self.sv_avg_b = short_b

        return self.W

    def get_support_vectors(self):

        return self.sv_X

    def prediction(self, input_data='', mode='test_data'):

        return super(SupportVectorRegression, self).prediction(input_data, mode)
