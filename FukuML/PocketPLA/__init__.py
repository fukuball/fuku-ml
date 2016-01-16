#encoding=utf8

import os
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
