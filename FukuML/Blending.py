#encoding=utf8

from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
import FukuML.LinearRegression as linear_regression


class BlendingClassifier(object):

    def add_model(self, model):

        self.models.append(model)

        return self.models

    @abstractmethod
    def prediction(self, input_data='', mode='test_data'):
        return

    def calculate_avg_error(self, input_data_file=''):

        data_num = 0
        error_num = 0
        avg_error = 0

        with open(input_data_file) as f:
            for line in f:
                data_num = data_num+1
                data = line.split()
                answer = data[-1]
                prediction = self.prediction(line)
                if float(prediction['prediction']) != float(answer):
                    error_num = error_num+1

        avg_error = float(error_num/data_num)

        return avg_error


class UniformBlendingClassifier(BlendingClassifier):

    def __init__(self):

        self.models = []

    def add_model(self, model):

        return super(UniformBlendingClassifier, self).add_model(model)

    def prediction(self, input_data='', mode='test_data'):

        prediction = {}
        vote = []

        for model in self.models:

            prediction = model.prediction(input_data, mode)
            vote.append(prediction['prediction'])

        prediction_return = max(set(vote), key=vote.count)

        if mode == 'future_data':
            data = input_data.split()
            input_data_x = [float(v) for v in data]
            input_data_x = np.ravel(input_data_x)
            return {"input_data_x": input_data_x, "input_data_y": None, "prediction": prediction_return}
        else:
            data = input_data.split()
            input_data_x = [float(v) for v in data[:-1]]
            input_data_x = np.ravel(input_data_x)
            input_data_y = float(data[-1])
            return {"input_data_x": input_data_x, "input_data_y": input_data_y, "prediction": prediction_return}

    def calculate_avg_error(self, input_data_file=''):

        return super(UniformBlendingClassifier, self).calculate_avg_error(input_data_file)


class LinearBlendingClassifier(BlendingClassifier):

    def __init__(self):

        self.models = []
        self.linear_mc = None

    def add_model(self, model):

        return super(LinearBlendingClassifier, self).add_model(model)

    def feature_transform(self, x_string):

        transform_x = []

        for model in self.models:
            prediction = model.prediction(x_string, 'future_data')
            transform_x.append(prediction['prediction'])

        return transform_x

    def train(self):

        predict_X = []

        for x in self.models[0].train_X:
            x_string = ' '.join(map(str, x.tolist()[1:]))
            transform_x = self.feature_transform(x_string)
            transform_x.insert(0, 1)
            predict_X.append(transform_x)

        self.linear_mc = linear_regression.MultiClassifier()
        self.linear_mc.load_train_data()
        self.linear_mc.load_test_data()
        self.linear_mc.train_X = np.array(predict_X)
        self.linear_mc.train_Y = self.models[0].train_Y
        self.linear_mc.set_param()
        self.linear_mc.init_W()
        self.linear_mc.train()

    def prediction(self, input_data='', mode='test_data'):

        transform_input_data = ''

        data = input_data.split()
        x_string = ' '.join(data[0:-1])
        transform_input_data = self.feature_transform(x_string)
        transform_input_data_string = ' '.join(map(str, transform_input_data))

        if mode == 'test_data':
            transform_input_data_string = transform_input_data_string + ' ' + data[-1]

        prediction = self.linear_mc.prediction(transform_input_data_string, mode)

        return prediction

    def calculate_avg_error(self, input_data_file=''):

        return super(LinearBlendingClassifier, self).calculate_avg_error(input_data_file)
