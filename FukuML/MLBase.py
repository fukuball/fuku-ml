#encoding=utf8

from abc import ABCMeta, abstractmethod


class Learner(object):
    __metaclass__ = ABCMeta

    status = 'empty'
    train_X = []
    train_Y = []
    W = []
    data_num = 0
    data_demension = 0
    tune_times = 0
    test_X = []
    test_Y = []

    @abstractmethod
    def __init__(self):
        return

    @abstractmethod
    def load_train_data(self, input_data_file=''):
        return

    @abstractmethod
    def load_test_data(self, input_data_file=''):
        return

    @abstractmethod
    def init_W(self):
        return

    @abstractmethod
    def score_function(self, x, W):
        return

    @abstractmethod
    def error_function(self, y_prediction, y_truth):
        return

    def calculate_avg_error(self, X, Y, W):

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

        '''
        Calculate test data avg error
        '''

        avg_error = self.calculate_avg_error(self.test_X, self.test_Y, self.W)

        return avg_error

    @abstractmethod
    def train(self):
        return

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
