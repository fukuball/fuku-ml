#encoding=utf8

import numpy as np


class DatasetLoader(object):

    status = 'empty'

    def __init__(self):

        self.status = 'empty'

    def load(self, input_data_file=''):

        X = []
        Y = []
        with open(input_data_file) as f:
            for line in f:
                data = line.split()
                x = [1] + [float(v) for v in data[:-1]]
                X.append(x)
                Y.append(float(data[-1]))

        return np.array(X), np.array(Y)
