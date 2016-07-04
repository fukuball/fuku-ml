#encoding=utf8

import os
import random
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml


class NeuralNetwork(ml.Learner):

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

        self.network_structure = []
        self.w_range_high = 0.1
        self.w_range_low = -0.1
        self.feed_mode = 'stochastic'
        self.step_eta = 0.01
        self.updates = 50000

    def load_train_data(self, input_data_file=''):

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/neural_network_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/neural_network_test.dat"))
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

    def set_param(self, network_structure=[8, 3, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000):

        self.network_structure = network_structure
        self.w_range_high = w_range_high
        self.w_range_low = w_range_low
        self.feed_mode = feed_mode
        self.step_eta = step_eta
        self.updates = updates

        return self.network_structure, self.w_range_high, self.w_range_low, self.feed_mode, self.step_eta, self.updates

    def init_W(self):

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.network_structure.insert(0, self.data_demension-1)

        for i in range(1, len(self.network_structure)):
            this_layer_w = np.random.uniform(self.w_range_low, self.w_range_high, (self.network_structure[i-1]+1, self.network_structure[i]))
            self.W.append(this_layer_w)

        return self.W

    def score_function(self, x, W):

        y_predict = x[1:]
        for i in range(0, len(W), 1):
            y_predict = np.tanh(np.dot(np.hstack((1, y_predict)), W[i]))

        score = y_predict[0]

        return score

    def error_function(self, y_prediction, y_truth):

        error = (y_prediction - y_truth) ** 2

        return error

    def calculate_avg_error(self, X, Y, W):

        return super(NeuralNetwork, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(NeuralNetwork, self).calculate_test_data_avg_error()

    def tanh_prime(self, s):
        tanh_prime_output = np.zeros(s.shape)
        for i in range(s.shape[0]):
            tanh_prime_output[i] = 4.0 / (np.exp(2*s[i])+np.exp(-2*s[i])+2)
        return tanh_prime_output

    def forward_process(self, x, y, W):
        forward_output = []
        pre_x = x
        for i in range(len(W)):
            pre_x = np.tanh(np.dot(pre_x, W[i]))
            forward_output.append(pre_x)
            pre_x = np.hstack((1, pre_x))
        return forward_output

    def backward_process(self, x, y, W, neuron_output):
        backward_output = []
        layer_num = len(neuron_output)
        score = np.dot(np.hstack((1, neuron_output[layer_num-2])), W[layer_num-1])
        error_gradient = np.array([-2 * (y-neuron_output[layer_num-1][0]) * self.tanh_prime(score)])
        #error_gradient = np.array([np.sum(-2 * (y - score) * np.hstack((1, neuron_output[layer_num-2])))])
        backward_output.insert(0, error_gradient)
        # Hidden layer
        for i in range(layer_num-2, -1, -1):
            if i == 0:
                score = np.dot(x, W[i])
            else:
                score = np.dot(np.hstack((1, neuron_output[i-1])), W[i])
            error_gradient = np.dot(error_gradient, W[i+1][1:].transpose()) * self.tanh_prime(score)
            backward_output.insert(0, error_gradient)
        return backward_output

    def update_W_with_gradient_decent(self, x, neuron_output, error_gradient):
        w_output = []
        layer_num = len(self.W)
        w_output.append(self.W[0] - self.step_eta * np.array([x]).transpose() * error_gradient[0])
        for i in range(1, layer_num, 1):
            w_output.append(self.W[i] - self.step_eta * np.array([np.hstack((1, neuron_output[i-1]))]).transpose() * error_gradient[i])
        return w_output

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        for i in range(0, self.updates):
            stochastic_i = random.randint(0, self.data_num-1)
            x = self.train_X[stochastic_i]
            y = self.train_Y[stochastic_i]
            neuron_output = self.forward_process(x, y, self.W)
            error_gradient = self.backward_process(x, y, self.W, neuron_output)
            self.W = self.update_W_with_gradient_decent(x, neuron_output, error_gradient)

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(NeuralNetwork, self).prediction(input_data, mode)


class BinaryClassifier(NeuralNetwork):

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

        self.network_structure = []
        self.w_range_high = 0.1
        self.w_range_low = -0.1
        self.feed_mode = 'stochastic'
        self.step_eta = 0.01
        self.updates = 50000

    def load_train_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_test_data(input_data_file)

    def set_param(self, network_structure=[8, 3, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000):

        return super(BinaryClassifier, self).set_param(network_structure, w_range_high, w_range_low, feed_mode, step_eta, updates)

    def init_W(self):

        return super(BinaryClassifier, self).init_W()

    def score_function(self, x, W):

        score = super(BinaryClassifier, self).score_function(x, W)
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

    def tanh_prime(self, s):

        return super(BinaryClassifier, self).tanh_prime(s)

    def forward_process(self, x, y, W):

        return super(BinaryClassifier, self).forward_process(x, y, W)

    def backward_process(self, x, y, W, neuron_output):

        return super(BinaryClassifier, self).backward_process(x, y, W, neuron_output)

    def update_W_with_gradient_decent(self, x, neuron_output, error_gradient):

        return super(BinaryClassifier, self).update_W_with_gradient_decent(x, neuron_output, error_gradient)

    def train(self):

        return super(BinaryClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)
