# encoding=utf8

import os
import collections
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml
import FukuML.DecisionTree as decision_tree
import FukuML.Blending as blending


class RandomForest(ml.Learner):

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

        self.learn_type = 'classifier'
        self.blending_model = None
        self.blending_model_num = 3

    def load_train_data(self, input_data_file=''):

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pocket_pla_binary_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file, data_type='string')

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pocket_pla_binary_test.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.test_X, self.test_Y

        self.test_X, self.test_Y = utility.DatasetLoader.load(input_data_file, data_type='string')

        return self.test_X, self.test_Y

    def set_param(self, learn_type='classifier', blending_model_num=3):

        self.learn_type = learn_type
        self.blending_model_num = blending_model_num

        return self.learn_type

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

        x_string = np.array(list(map(str, x)))
        x_string = ' '.join(x_string[:])
        prediction = self.blending_model.prediction(x_string, 'future_data')
        score = prediction['prediction']

        return score

    def error_function(self, y_prediction, y_truth):

        if self.learn_type == 'classifier':
            if float(y_prediction) != float(y_truth):
                return 1
            else:
                return 0
        elif self.learn_type == 'regression':
            return (float(y_prediction) - float(y_truth)) ** 2

    def calculate_avg_error(self, X, Y, W):

        return super(RandomForest, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(RandomForest, self).calculate_test_data_avg_error()

    def train(self):

        if self.learn_type == 'classifier':
            self.blending_model = blending.UniformBlendingClassifier()
        elif self.learn_type == 'regression':
            self.blending_model = blending.UniformBlendingRegression()

        for i in range(self.blending_model_num):
            dt = decision_tree.CART()
            dt.load_train_data()
            dt.train_X, dt.train_Y = utility.DatasetLoader.bootstrap_bagging(self.train_X, self.train_Y, self.data_num)
            dt.set_param(self.learn_type)
            dt.init_W()
            dt.train()
            self.blending_model.add_model(dt)

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return self.blending_model.prediction(input_data, mode)
