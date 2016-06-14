#encoding=utf8

import os
import numpy as np
import FukuML.Utility as utility
import FukuML.SupportVectorMachine as svm
import FukuML.LogisticRegression as logistic_regression
import FukuML.MLBase as ml


class ProbabilisticSVM(ml.Learner):

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
        self.svm_kernel = 'soft_gaussian_kernel'
        self.gamma = 1
        self.C = 0.1

        self.svm_processor = ''
        self.logistic_processor = ''

    def load_train_data(self, input_data_file=''):

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

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/logistic_regression_test.dat"))
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

    def set_param(self, feed_mode='batch', step_eta=0.126, updates=2000, C=0.1):

        self.feed_mode = feed_mode
        self.step_eta = step_eta
        self.updates = updates
        self.C = C

        return self.feed_mode, self.step_eta, self.updates, self.C

    def init_W(self, mode='normal'):

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.W = np.zeros(self.data_demension)

        return self.W

    def score_function(self, x, W):

        svm_process_x = self.svm_score(x)
        svm_process_x = [1] + [svm_process_x]

        score = self.logistic_processor.theta(np.inner(svm_process_x, self.logistic_processor.W))

        return score

    def error_function(self, x, y, W):

        svm_process_x = self.svm_score(x)
        svm_process_x = [1] + [svm_process_x]

        error = np.log(1 + np.exp((-1)*y*np.inner(svm_process_x, self.logistic_processor.W)))

        return error

    def calculate_avg_error(self, X, Y, W):

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(X[i], Y[i], W)

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        avg_error = self.calculate_avg_error(self.test_X, self.test_Y, self.W)

        return avg_error

    def svm_score(self, x):

        x = x[1:]

        '''
        original_X = self.svm_processor.train_X[:, 1:]
        score = 0
        for i in range(len(self.svm_processor.sv_alpha)):
            score += self.svm_processor.sv_alpha[i] * self.svm_processor.sv_Y[i] * utility.Kernel.gaussian_kernel(self, original_X[self.svm_processor.sv_index[i]], x)
        score = score + self.svm_processor.sv_avg_b
        '''

        score = np.sum(self.svm_processor.sv_alpha * self.svm_processor.sv_Y * utility.Kernel.kernel_matrix_xX(self, x, self.svm_processor.sv_X)) + self.svm_processor.sv_avg_b

        return score

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        self.svm_processor = svm.BinaryClassifier()
        self.svm_processor.load_train_data()
        self.svm_processor.train_X = self.train_X
        self.svm_processor.train_Y = self.train_Y
        self.svm_processor.set_param(svm_kernel=self.svm_kernel, gamma=self.gamma, C=self.C)
        self.svm_processor.init_W()
        self.svm_processor.train()

        # slow
        svm_transform_X = np.apply_along_axis(self.svm_score, axis=1, arr=self.train_X)
        svm_transform_X = np.reshape(svm_transform_X, (-1, 1))
        svm_transform_X0 = np.reshape(np.ones(self.data_num), (-1, 1))
        svm_transform_X = np.concatenate((svm_transform_X0, svm_transform_X), axis=1)

        self.logistic_processor = logistic_regression.LogisticRegression()
        self.logistic_processor.load_train_data()
        self.logistic_processor.train_X = svm_transform_X
        self.logistic_processor.train_Y = self.train_Y
        self.logistic_processor.set_param(feed_mode=self.feed_mode, step_eta=self.step_eta, updates=self.updates)
        self.logistic_processor.init_W()
        self.logistic_processor.train()

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(ProbabilisticSVM, self).prediction(input_data, mode)
