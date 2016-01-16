#encoding=utf8

import os
import random
import numpy as np

__version__ = '0.0.1'
__all__ = ['Machine Learning', 'Pocket Perceptron Learning Algorithm', 'Pocket PLA']

status = 'empty'
train_X = []
train_Y = []
W = []
data_num = 0
data_demension = 0
tune_times = 0
test_data_x = []
test_data_y = 0
test_X = []
test_Y = []
avg_error = float('Inf')
put_in_pocket_times = 0


def load_train_data(input_data_file=''):

    '''
    Load train data
    Please check dataset/pocket_pla_train.dat to understand the data format
    Each feature of data x separated with spaces
    And the ground truth y put in the end of line separated by a space
    '''

    global status
    global train_X
    global train_Y

    status = 'load_train_data'

    if (input_data_file == ''):
        input_data_file = os.path.normpath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), os.pardir), "dataset/pocket_pla_train.dat"))
    else:
        if (os.path.isfile(input_data_file) is not True):
            print("Please make sure input_data_file path is correct.")
            return train_X, train_Y

    X = []
    Y = []
    with open(input_data_file) as f:
        for line in f:
            data = line.split()
            x = [1] + [float(v) for v in data[:-1]]
            X.append(x)
            Y.append(float(data[-1]))

    train_X = np.array(X)
    train_Y = np.array(Y)

    return train_X, train_Y


def load_test_data(input_data_file=''):

    '''
    Load test data
    Please check dataset/pocket_pla_test.dat to understand the data format
    Each feature of data x separated with spaces
    And the ground truth y put in the end of line separated by a space
    '''

    global test_X
    global test_Y

    if (input_data_file == ''):
        input_data_file = os.path.normpath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), os.pardir), "dataset/pocket_pla_test.dat"))
    else:
        if (os.path.isfile(input_data_file) is not True):
            print("Please make sure input_data_file path is correct.")
            return test_X, test_Y

    X = []
    Y = []
    with open(input_data_file) as f:
        for line in f:
            data = line.split()
            x = [1] + [float(v) for v in data[:-1]]
            X.append(x)
            Y.append(float(data[-1]))

    test_X = np.array(X)
    test_Y = np.array(Y)

    return test_X, test_Y


def init_W():

    '''
    Init the W
    Simple way is init W all zeros
    '''

    global status
    global train_X
    global train_Y
    global W
    global data_num
    global data_demension

    if (status != 'load_train_data') and (status != 'train'):
        print("Please load train data first.")
        return W

    status = 'init'

    data_num = len(train_Y)
    data_demension = len(train_X[0])
    W = np.zeros(data_demension)

    return W


def calculate_avg_error(X, Y, W):

    '''
    Get current avg error from X, Y, W
    '''

    data_num = len(Y)
    error_num = sum([1 for i in range(data_num) if np.sign(np.inner(X[i], W)) != Y[i]])
    avg_error = error_num / float(data_num)

    return avg_error


def train(updates=50, mode='random', alpha=1):

    '''
    Train Pocket Perceptron Learning Algorithm
    From f(x) = WX
    Find best h(x) = WX similar to f(x)
    Output W
    '''

    global status
    global train_X
    global train_Y
    global W
    global tune_times
    global data_num
    global avg_error
    global put_in_pocket_times

    if (status != 'init'):
        print("Please load train data and init W first.")
        return W

    status = 'train'

    new_W = W

    avg_error = calculate_avg_error(train_X, train_Y, new_W)

    for _ in range(updates):
        if (mode is 'naive_cycle'):
            data_check_order = range(data_num)
        elif (mode is 'random'):
            data_check_order = range(data_num)
            data_check_order = random.sample(data_check_order, data_num)
        else:
            data_check_order = range(data_num)
            data_check_order = random.sample(data_check_order, data_num)
        for i in data_check_order:
            if np.sign(np.inner(train_X[i], new_W)) != train_Y[i]:
                tune_times += 1
                new_W = new_W + alpha * (train_Y[i] * train_X[i])
                new_avg_error = calculate_avg_error(train_X, train_Y, new_W)
                if new_avg_error < avg_error:
                    put_in_pocket_times += 1
                    avg_error = new_avg_error
                    W = new_W
                break

    return W


def prediction(test_data=''):

    '''
    Make prediction
    input test data
    output the prediction
    '''

    global status
    global W
    global test_data_x
    global test_data_y

    prediction = 0

    if (status != 'train'):
        print("Please load train data and init W then train the W first.")
        return prediction

    if (test_data == ''):
        print("Please input test data for prediction.")
        return prediction

    data = test_data.split()
    test_data_x = [1] + [float(v) for v in data[:-1]]
    test_data_y = float(data[-1])

    prediction = np.sign(np.dot(test_data_x, W))

    return prediction
