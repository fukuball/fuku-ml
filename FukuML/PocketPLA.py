#encoding=utf8

import os
import random
import numpy as np
import FukuML.Utility as utility


class BinaryClassifier(object):

    status = 'empty'
    train_X = []
    train_Y = []
    W = []
    data_num = 0
    data_demension = 0
    tune_times = 0
    test_X = []
    test_Y = []

    temp_avg_error = float('Inf')
    put_in_pocket_times = 0

    def __init__(self):

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = []
        self.data_num = 0
        self.data_demension = 0
        self.tune_times = 0
        self.test_X = []
        self.test_Y = []

        self.temp_avg_error = float('Inf')
        self.put_in_pocket_times = 0

    def load_train_data(self, input_data_file=''):

        '''
        Load train data
        Please check dataset/pocket_pla_binary_train.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pocket_pla_binary_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        dataset_loader = utility.DatasetLoader()
        self.train_X, self.train_Y = dataset_loader.load(input_data_file)

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

        dataset_loader = utility.DatasetLoader()
        self.test_X, self.test_Y = dataset_loader.load(input_data_file)

        return self.test_X, self.test_Y

    def init_W(self):

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
        # need refector

        '''
        Calculate avg error from X, Y, W
        '''

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(self.score_function(X[i], W), Y[i])

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):
        # need refector

        '''
        Calculate test data avg error
        '''

        avg_error = self.calculate_avg_error(self.test_X, self.test_Y, self.W)

        return avg_error

    def train(self, updates=50, mode='random', alpha=1):

        '''
        Train Pocket Perceptron Learning Algorithm
        From f(x) = WX
        Find best h(x) = WX similar to f(x)
        Output W
        '''

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        new_W = self.W

        self.temp_avg_error = self.calculate_avg_error(self.train_X, self.train_Y, new_W)

        for _ in range(updates):
            if (mode is 'naive_cycle'):
                data_check_order = range(self.data_num)
            elif (mode is 'random'):
                data_check_order = range(self.data_num)
                data_check_order = random.sample(data_check_order, self.data_num)
            else:
                data_check_order = range(self.data_num)
                data_check_order = random.sample(data_check_order, self.data_num)
            for i in data_check_order:

                if self.error_function(self.score_function(self.train_X[i], new_W), self.train_Y[i]):
                    self.tune_times += 1
                    new_W = new_W + alpha * (self.train_Y[i] * self.train_X[i])
                    new_avg_error = self.calculate_avg_error(self.train_X, self.train_Y, new_W)
                    if new_avg_error < self.temp_avg_error:
                        self.put_in_pocket_times += 1
                        self.temp_avg_error = new_avg_error
                        self.W = new_W
                    break

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        '''
        Make prediction
        input test data
        output the prediction
        '''

        prediction = 0

        if (self.status != 'train'):
            print("Please load train data and init W then train the W first.")
            return prediction

        if (input_data == ''):
            print("Please input test data for prediction.")
            return prediction

        if mode == 'future_data':
            data = input_data.split()
            input_data_x = [1] + [float(v) for v in data]
            prediction = self.score_function(input_data_x, self.W)
            return {"input_data_x": input_data_x, "input_data_y": None, "prediction": prediction}
        else:
            data = input_data.split()
            input_data_x = [1] + [float(v) for v in data[:-1]]
            input_data_y = float(data[-1])
            prediction = self.score_function(input_data_x, self.W)
            return {"input_data_x": input_data_x, "input_data_y": input_data_y, "prediction": prediction}
