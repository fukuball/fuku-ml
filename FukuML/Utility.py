#encoding=utf8

import numpy as np
import pickle


class DatasetLoader(object):

    @staticmethod
    def load(input_data_file=''):
        """load file"""
        X = []
        Y = []
        with open(input_data_file) as f:
            for line in f:
                data = line.split()
                x = [1] + [float(v) for v in data[:-1]]
                X.append(x)
                Y.append(float(data[-1]))

        return np.array(X), np.array(Y)


class Serializer(object):

    @staticmethod
    def serialize(need_serialize=''):
        """serialize"""
        return pickle.dumps(need_serialize, protocol=0)

    @staticmethod
    def deserialize(pickle_serialized=''):
        """ddserialize"""
        return pickle.loads(pickle_serialized)
