#encoding=utf8

import os,sys
import random
from numpy import array, inner, zeros, sign

train_X = []
train_Y = []

def load_train_data():

    global train_X
    global train_Y

    input_file = os.path.normpath( os.path.join( os.path.join( os.path.join( os.getcwd(), os.path.dirname(__file__) ), os.pardir ) , "dataset/pla_train.dat" ) )

    X = []
    Y = []
    with open(input_file) as f:
        for line in f:
            data = line.split()
            x = [1] + [float(v) for v in data[:-1]]
            X.append(x)
            Y.append(int(data[-1]))
    train_X = array(X)
    train_Y = array(Y)
    return train_X, train_Y

def train():

    global train_X
    global train_Y

    data_num = len(train_Y)
    data_demension = len(train_X[0])
    W = zeros(data_demension)

    idx = range(data_num)

    tune_times = 0
    point_wise_i = 0
    flag = True

    while True:
        if point_wise_i == data_num:
            if flag:
                break
            point_wise_i = 0
            flag = True

        if sign(inner(train_X[point_wise_i], W)) != train_Y[point_wise_i]:
            flag = False
            tune_times += 1
            W = W + train_Y[point_wise_i] * train_X[point_wise_i]
        point_wise_i += 1

    return W
