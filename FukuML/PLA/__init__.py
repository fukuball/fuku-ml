#encoding=utf8

import os
import random
import numpy as np

__version__ = '0.0.1'
__all__ = ['Machine Learning', 'Perceptron Learning Algorithm', 'PLA']

status = 'empty'
train_X = []
train_Y = []
W = []
data_num = 0
data_demension = 0
tune_times = 0
test_data_x = []
test_data_y = 0


def load_train_data(input_data_file=''):

    '''
    Load train data
    Please check dataset/pla_train.dat to understand the data format
    Each feature of data x separated with spaces
    And the ground truth y put in the end of line separated by a space
    '''

    global status
    global train_X
    global train_Y

    status = 'load_train_data'

    if (input_data_file == ''):
        input_data_file = os.path.normpath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), os.pardir), "dataset/pla_train.dat"))
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


def train(mode='naive_cycle', alpha=1):

    '''
    Train Perceptron Learning Algorithm
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

    if (status != 'init'):
        print("Please load train data and init W first.")
        return W

    status = 'train'

    if (mode is 'random'):
        data_check_order = range(data_num)
        data_check_order = random.sample(data_check_order, data_num)
    elif (mode is 'naive_cycle'):
        data_check_order = range(data_num)
    else:
        data_check_order = range(data_num)

    tune_times = 0
    k = 0
    flag = True

    while True:
        if k == data_num:
            if flag:
                break
            k = 0
            flag = True

        point_wise_i = data_check_order[k]
        if np.sign(np.inner(train_X[point_wise_i], W)) != train_Y[point_wise_i]:
            flag = False
            tune_times += 1
            W = W + alpha * (train_Y[point_wise_i] * train_X[point_wise_i])
        k += 1

    return W


def prediction(test_data=''):

    global status
    global W
    global test_data_x
    global test_data_y

    prediction = 0

    if (status != 'train'):
        print("Please load train data and init W then train the W first.")
        return prediction

    if (test_data == ''):
        print("Please input test dat for prediction.")
        return prediction

    data = test_data.split()
    test_data_x = [1] + [float(v) for v in data[:-1]]
    test_data_y = float(data[-1])

    prediction = np.sign(np.dot(test_data_x, W))

    return prediction
