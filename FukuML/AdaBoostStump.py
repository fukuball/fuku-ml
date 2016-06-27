#encoding=utf8

import os
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml
import FukuML.DecisionStump as decision_stump


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

        self.run_t = 40
        self.weak_learner = []
        self.alpha = []
        self.temp_train_X = []

    def load_train_data(self, input_data_file=''):

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/decision_stump_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/decision_stump_test.dat"))
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
        self.weak_learner = [None] * self.run_t
        self.alpha = [0.0] * self.run_t
        self.W = np.zeros(self.data_demension)

        return self.W

    def score_function(self, x, W):

        score = 0.0

        for i, weak_learner in enumerate(self.weak_learner):
            predict_string = np.array(list(map(str, x)))
            predict_string = ' '.join(predict_string[1:])
            prediction = weak_learner.prediction(predict_string, 'future_data')
            score = score + (self.alpha[i] * prediction['prediction'])

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

    def calculate_alpha_u(self, weak_learner, u):

        alpha = 0.0
        epsiloin = 0.0
        data_num = len(weak_learner.train_Y)

        for i in range(data_num):
            predict_string = np.array(list(map(str, weak_learner.train_X[i])))
            predict_string = ' '.join(predict_string[1:]) + ' ' + str(weak_learner.train_Y[i])
            prediction = weak_learner.prediction(predict_string, 'test_data')
            if (float(prediction['prediction']) != float(prediction['input_data_y'])):
                epsiloin += (u[i] * 1.0)

        epsiloin = epsiloin / np.sum(u)
        tune_alpha = np.sqrt((1.0-epsiloin)/epsiloin)
        alpha = np.log(tune_alpha)

        new_u = []

        for i in range(data_num):
            predict_string = np.array(list(map(str, weak_learner.train_X[i])))
            predict_string = ' '.join(predict_string[1:]) + ' ' + str(weak_learner.train_Y[i])
            prediction = weak_learner.prediction(predict_string, 'test_data')
            if (float(prediction['prediction']) != float(prediction['input_data_y'])):
                new_u.append(u[i] * tune_alpha)
            else:
                new_u.append(u[i] / tune_alpha)

        return alpha, np.array(new_u)

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        u = np.array([(1.0 / self.data_num)] * self.data_num)

        for t in range(self.run_t):

            #np.random.choice(np.arange(self.data_num), self.data_num, p=(u/sum(u)))

            print("Round "+str(t+1))

            decision_stump_bc = decision_stump.BinaryClassifier()
            decision_stump_bc.status = 'load_train_data'
            decision_stump_bc.train_X = self.train_X
            decision_stump_bc.train_Y = self.train_Y
            decision_stump_bc.set_param(u)
            decision_stump_bc.init_W()
            decision_stump_bc.train()

            alpha, u = self.calculate_alpha_u(decision_stump_bc, u)

            self.weak_learner[t] = decision_stump_bc
            self.alpha[t] = alpha

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
