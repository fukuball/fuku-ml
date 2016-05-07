#encoding=utf8

import os
import random
import operator
import itertools
#import collections
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml
import FukuML.LinearRegression as linear_regression
#np.set_printoptions(threshold=np.nan)


class LogisticRegression(ml.Learner):

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
        self.step_ita = 0.126
        self.updates = 2000

    def load_train_data(self, input_data_file=''):

        '''
        Load train data
        Please check dataset/logistic_regression_train.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

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

        '''
        Load test data
        Please check dataset/logistic_regression_test.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

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

    def set_param(self, feed_mode='batch', step_ita=0.126, updates=2000):

        self.feed_mode = feed_mode
        self.step_ita = step_ita
        self.updates = updates

        return self.feed_mode, self.step_ita, self.updates

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

    def theta(self, s):

        '''
        Theta sigmoid function
        '''

        s = np.where(s < -709, -709, s)

        return 1/(1 + np.exp((-1)*s))

    def score_function(self, x, W):
        # need refector

        '''
        Score function to calculate score
        '''

        score = self.theta(np.inner(x, W))

        return score

    def error_function(self, x, y, W):
        # need refector

        '''
        Error function to calculate error: cross entropy error
        '''

        error = np.log(1 + np.exp((-1)*y*np.inner(x, W)))

        return error

    def calculate_gradient(self, X, Y, W):

        if type(Y) is np.ndarray:
            data_num = len(Y)
        else:
            data_num = 1

        gradient_average = np.dot(self.theta((-1)*Y*np.dot(W, X.transpose()))*((-1)*Y), X)/data_num

        return gradient_average

    def calculate_avg_error(self, X, Y, W):

        '''
        Calculate avg error from X, Y, W
        '''

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(X[i], Y[i], W)

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        '''
        Calculate test data avg error
        '''

        avg_error = self.calculate_avg_error(self.test_X, self.test_Y, self.W)

        return avg_error

    def train(self):

        '''
        Train Linear Regression Algorithm
        From f(x) = WX
        Find best h(x) = WX similar to f(x)
        Output W

        mode: batch / stochastic
        '''

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        for i in range(0, self.updates):
            if self.feed_mode == 'stochastic':
                stochastic_i = random.randint(0, self.data_num-1)
                x = self.train_X[stochastic_i]
                y = self.train_Y[stochastic_i]
                gradient = self.calculate_gradient(x, y, self.W)
            else:
                gradient = self.calculate_gradient(self.train_X, self.train_Y, self.W)
            if np.linalg.norm(gradient) == 0:
                return self.W
            self.W = self.W - self.step_ita * gradient

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(LogisticRegression, self).prediction(input_data, mode)


class BinaryClassifier(LogisticRegression):

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
        self.step_ita = 0.126
        self.updates = 2000

    def load_train_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_test_data(input_data_file)

    def set_param(self, feed_mode='batch', step_ita=0.126, updates=2000):

        return super(BinaryClassifier, self).set_param(feed_mode, step_ita, updates)

    def init_W(self, mode='normal'):

        return super(BinaryClassifier, self).init_W(mode)

    def theta(self, s):

        return super(BinaryClassifier, self).theta(s)

    def score_function(self, x, W):

        '''
        Score function to calculate score
        '''

        score = super(BinaryClassifier, self).score_function(x, W)
        if score >= 0.5:
            score = 1.0
        else:
            score = -1.0

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

    def calculate_gradient(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_gradient(X, Y, W)

    def calculate_avg_error(self, X, Y, W):

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(self.score_function(X[i], W), Y[i])

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        return super(BinaryClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)


class MultiClassifier(LogisticRegression):

    class_list = []
    temp_train_X = []
    temp_train_Y = []
    temp_W = {}
    temp_data_num = 0
    decomposition = 'ova'

    def __init__(self):

        """init"""

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = {}
        self.data_num = 0
        self.data_demension = 0
        self.test_X = []
        self.test_Y = []
        self.feature_transform_mode = ''
        self.feature_transform_degree = 1

        self.feed_mode = 'batch'
        self.step_ita = 0.126
        self.updates = 2000
        self.class_list = []
        self.temp_train_X = []
        self.temp_train_Y = []
        self.temp_W = {}
        self.temp_data_num = 0
        self.decomposition = 'ova'

    def load_train_data(self, input_data_file=''):

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/digits_multiclass_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/digits_multiclass_test.dat"))
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

    def set_param(self, feed_mode='batch', step_ita=0.126, updates=2000):

        return super(MultiClassifier, self).set_param(feed_mode, step_ita, updates)

    def init_W(self, mode='normal', decomposition='ova'):

        self.W = {}

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.decomposition = decomposition

        if self.decomposition == 'ovo':
            self.class_list = list(itertools.combinations(np.unique(self.train_Y), 2))
        elif self.decomposition == 'ova':
            self.class_list = np.unique(self.train_Y)

        for class_item in self.class_list:
            self.W[class_item] = np.zeros(self.data_demension)

        if mode == 'linear_regression_accelerator':
            accelerator = linear_regression.Accelerator()
            for class_item in self.class_list:
                if self.decomposition == 'ovo':
                    modify_X, modify_Y = self.modify_XY(self.train_X, self.train_Y, class_item)
                    self.temp_train_X = self.train_X
                    self.temp_train_Y = self.train_Y
                    self.train_X = modify_X
                    self.train_Y = modify_Y
                    self.temp_data_num = self.data_num
                    self.data_num = len(self.train_Y)
                    self.temp_W = self.W
                    self.W = self.temp_W[class_item]
                    self.temp_W[class_item] = accelerator.init_W(self)
                    self.train_X = self.temp_train_X
                    self.train_Y = self.temp_train_Y
                    self.temp_train_X = []
                    self.temp_train_Y = []
                    self.data_num = self.temp_data_num
                    self.temp_data_num = 0
                    self.W = self.temp_W
                    self.temp_W = {}
                elif self.decomposition == 'ova':
                    modify_Y = self.modify_Y(self.train_Y, class_item)
                    self.temp_train_Y = self.train_Y
                    self.train_Y = modify_Y
                    self.temp_W = self.W
                    self.W = self.temp_W[class_item]
                    self.temp_W[class_item] = accelerator.init_W(self)
                    self.train_Y = self.temp_train_Y
                    self.temp_train_Y = []
                    self.W = self.temp_W
                    self.temp_W = {}

        return self.W

    def theta(self, s):

        return super(MultiClassifier, self).theta(s)

    def score_function(self, x, W):

        return super(MultiClassifier, self).score_function(x, W)

    def score_function_all_class(self, x, W):

        score_list = {}
        ovo_vote = []

        for class_item in self.class_list:
            score = self.score_function(x, W[class_item])
            if self.decomposition == 'ovo':
                if score >= 0.5:
                    score_list[class_item] = class_item[0]
                else:
                    score_list[class_item] = class_item[1]
                ovo_vote.append(score_list[class_item])
            elif self.decomposition == 'ova':
                score_list[class_item] = score

        if self.decomposition == 'ovo':
            return max(set(ovo_vote), key=ovo_vote.count)
        elif self.decomposition == 'ova':
            return max(score_list.items(), key=operator.itemgetter(1))[0]

    def error_function(self, y_prediction, y_truth):

        if y_prediction != y_truth:
            return 1
        else:
            return 0

    def calculate_gradient(self, X, Y, W):

        return super(MultiClassifier, self).calculate_gradient(X, Y, W)

    def calculate_avg_error(self, X, Y, W):

        return super(MultiClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_avg_error_all_class(self, X, Y, W):

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(self.score_function_all_class(X[i], W), Y[i])

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        return super(MultiClassifier, self).calculate_test_data_avg_error()

    def modify_Y(self, Y, class_item):

        modify_Y = []
        for yi in Y:
            if yi == class_item:
                modify_Y.append(1)
            else:
                modify_Y.append(-1)

        return np.array(modify_Y)

    def modify_XY(self, X, Y, class_item):

        modify_X = []
        modify_Y = []

        for idx, val in enumerate(Y):
            if val == class_item[0]:
                modify_Y.append(1)
                modify_X.append(X[idx])
            elif val == class_item[1]:
                modify_Y.append(-1)
                modify_X.append(X[idx])

        return np.array(modify_X), np.array(modify_Y)

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        for class_item in self.class_list:
            self.status = 'init'
            if self.decomposition == 'ovo':
                modify_X, modify_Y = self.modify_XY(self.train_X, self.train_Y, class_item)
                self.temp_train_X = self.train_X
                self.temp_train_Y = self.train_Y
                self.train_X = modify_X
                self.train_Y = modify_Y
                self.temp_data_num = self.data_num
                self.data_num = len(self.train_Y)
                self.temp_W = self.W
                self.W = self.temp_W[class_item]
                self.temp_W[class_item] = super(MultiClassifier, self).train()
                self.train_X = self.temp_train_X
                self.train_Y = self.temp_train_Y
                self.temp_train_X = []
                self.temp_train_Y = []
                self.data_num = self.temp_data_num
                self.temp_data_num = 0
                self.W = self.temp_W
                self.temp_W = {}
                print("class %d to %d learned." % (class_item[0], class_item[1]))
            elif self.decomposition == 'ova':
                modify_Y = self.modify_Y(self.train_Y, class_item)
                self.temp_train_Y = self.train_Y
                self.train_Y = modify_Y
                self.temp_W = self.W
                self.W = self.temp_W[class_item]
                self.temp_W[class_item] = super(MultiClassifier, self).train()
                self.train_Y = self.temp_train_Y
                self.temp_train_Y = []
                self.W = self.temp_W
                self.temp_W = {}
                print("class %d learned." % class_item)

        self.status = 'train'

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        prediction = {}
        prediction_list = {}
        prediction_return = 0.0
        ovo_vote = []

        for class_item in self.class_list:
            self.temp_W = self.W
            self.W = self.temp_W[class_item]
            prediction = super(MultiClassifier, self).prediction(input_data, mode)
            if self.decomposition == 'ovo':
                if prediction['prediction'] >= 0.5:
                    prediction_list[class_item] = class_item[0]
                else:
                    prediction_list[class_item] = class_item[1]
                ovo_vote.append(prediction_list[class_item])
            elif self.decomposition == 'ova':
                prediction_list[class_item] = prediction['prediction']
            self.W = self.temp_W
            self.temp_W = {}

        if self.decomposition == 'ovo':
            #counter = collections.Counter(ovo_vote)
            prediction_return = max(set(ovo_vote), key=ovo_vote.count)
        elif self.decomposition == 'ova':
            prediction_return = max(prediction_list.items(), key=operator.itemgetter(1))[0]

        return {
            "input_data_x": prediction['input_data_x'],
            "input_data_y": prediction['input_data_y'],
            "prediction": prediction_return,
            "prediction_list": prediction_list,
        }
