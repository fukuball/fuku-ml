#encoding=utf8

import os
import numpy as np
import cvxopt
import cvxopt.solvers
import FukuML.Utility as utility
import FukuML.MLBase as ml
import FukuML.LinearRegression as linear_regression


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

        self.svm_kernel = 'primal_hard_margin'

    def load_train_data(self, input_data_file=''):

        '''
        Load train data
        Please check dataset/pla_binary_train.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pla_binary_train.dat"))
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

    def setParam(self, svm_kernel='primal_hard_margin'):

        self.svm_kernel = svm_kernel

        return self.svm_kernel

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

        if mode == 'linear_regression_accelerator':
            accelerator = linear_regression.Accelerator()
            self.W = accelerator.init_W(self)

        return self.W

    def score_function(self, x, W):
        # need refector

        '''
        Score function to calculate score
        '''

        score = np.sign(np.inner(x, W))

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

        return super(BinaryClassifier, self).calculate_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        # P = Q, q = p, G = -A, h = -c

        if (self.svm_kernel != 'primal_hard_margin'):
            eye_process = np.eye(self.data_demension)
            eye_process[0][0] = 0
            P = cvxopt.matrix(eye_process)
            q = cvxopt.matrix(np.zeros(self.data_demension))
            G = cvxopt.matrix(np.reshape(self.train_Y, (-1, 1)) * self.train_X * -1)
            h = cvxopt.matrix(np.ones(self.data_num) * -1)
        else:
            eye_process = np.eye(self.data_demension)
            eye_process[0][0] = 0
            P = cvxopt.matrix(eye_process)
            q = cvxopt.matrix(np.zeros(self.data_demension))
            G = cvxopt.matrix(np.reshape(self.train_Y, (-1, 1)) * self.train_X * -1)
            h = cvxopt.matrix(np.ones(self.data_num) * -1)

        cvxopt.solvers.options['show_progress'] = False

        solution = cvxopt.solvers.qp(P, q, G, h)
        self.W = np.array(solution['x'])
        self.W = np.ravel(self.W)

        return self.W

    def getMarge(self):

        return 1/np.linalg.norm(self.W[1:])

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
