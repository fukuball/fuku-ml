#encoding=utf8

import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import legendre


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

    @staticmethod
    def featureTransform(X, mode='polynomial', degree=1):

        poly = PolynomialFeatures(degree)
        process_X = poly.fit_transform(X)

        if mode == 'legendre':
            lege = legendre(degree)
            process_X = lege(process_X)

        return process_X


class Serializer(object):

    @staticmethod
    def serialize(need_serialize=''):
        """serialize"""
        return pickle.dumps(need_serialize, protocol=0)

    @staticmethod
    def deserialize(pickle_serialized=''):
        """ddserialize"""
        return pickle.loads(pickle_serialized)
