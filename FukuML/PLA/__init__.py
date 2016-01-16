#encoding=utf8

import os
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
            Y.append(int(data[-1]))

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


def train():

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

    tune_times = 0
    point_wise_i = 0
    flag = True

    while True:
        if point_wise_i == data_num:
            if flag:
                break
            point_wise_i = 0
            flag = True

        if np.sign(np.inner(train_X[point_wise_i], W)) != train_Y[point_wise_i]:
            flag = False
            tune_times += 1
            W = W + train_Y[point_wise_i] * train_X[point_wise_i]
        point_wise_i += 1

    return W
