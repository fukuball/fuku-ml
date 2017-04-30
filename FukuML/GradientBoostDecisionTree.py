# encoding=utf8

import os
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml
import FukuML.DecisionTree as decision_tree
import FukuML.LinearRegression as linear_regression


class Regression(ml.Learner):

    # too slow for high dimension data, can't do digits multi classifier
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

        self.run_t = 40
        self.decision_tree = []
        self.alpha = []

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

    def set_param(self, run_t):

        self.run_t = run_t

        return self.run_t

    def init_W(self, mode='normal'):

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.decision_tree = [None] * self.run_t
        self.alpha = [0.0] * self.run_t
        self.W = np.zeros(self.data_demension)

        return self.W

    def score_function(self, x, W):

        score = 0.0

        for i, weak_learner in enumerate(self.decision_tree):
            predict_string = np.array(list(map(str, x)))
            predict_string = ' '.join(predict_string[1:])
            prediction = weak_learner.prediction(predict_string, 'future_data')
            score = score + (self.alpha[i] * prediction['prediction'])

        return score

    def error_function(self, y_prediction, y_truth):

        error = (y_prediction - y_truth) ** 2

        return error

    def calculate_avg_error(self, X, Y, W):

        return super(Regression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(Regression, self).calculate_test_data_avg_error()

    def calculate_alpha_s(self, weak_learner, s):

        alpha = 0.0
        new_s = s

        data_num = len(weak_learner.train_Y)
        X = []

        for i in range(data_num):
            predict_string = np.array(list(map(str, weak_learner.train_X[i])))
            predict_string = ' '.join(predict_string[1:]) + ' ' + str(weak_learner.train_Y[i])
            prediction = weak_learner.prediction(predict_string, 'test_data')
            X.append([float(prediction['prediction'])])

        X = np.array(X)

        linear = linear_regression.LinearRegression()
        linear.status = 'load_train_data'
        linear.train_X = X
        linear.train_Y = weak_learner.train_Y - s
        linear.set_param()
        linear.init_W()
        linear.train()

        alpha = linear.W[0]
        new_s = s + alpha * np.ravel(X)

        return alpha, new_s

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        s = np.array([0] * self.data_num)

        for t in range(self.run_t):

            # np.random.choice(np.arange(self.data_num), self.data_num, p=(u/sum(u)))

            print("Round " + str(t + 1))

            decision_tree_c = decision_tree.CART()
            decision_tree_c.status = 'load_train_data'
            decision_tree_c.train_X = self.train_X
            decision_tree_c.train_Y = self.train_Y - s
            decision_tree_c.set_param(learn_type='regression', tree_height_limit=3)
            decision_tree_c.init_W()
            decision_tree_c.train()

            alpha, s = self.calculate_alpha_s(decision_tree_c, s)

            self.decision_tree[t] = decision_tree_c
            self.alpha[t] = alpha

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(Regression, self).prediction(input_data, mode)
