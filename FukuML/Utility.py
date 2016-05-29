#encoding=utf8

import numpy as np
import pickle
import math
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
    def feature_transform(X, mode='polynomial', degree=1):

        poly = PolynomialFeatures(degree)
        process_X = poly.fit_transform(X)

        if mode == 'legendre':
            lege = legendre(degree)
            process_X = lege(process_X)

        return process_X

    @staticmethod
    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    @staticmethod
    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1

        return X1, y1, X2, y2

    @staticmethod
    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


class CrossValidator(object):

    def __init__(self):

        """init"""

        self.models = []
        self.avg_errors = []

    def add_model(self, model):

        self.models.append(model)

        return self.models

    def excute(self):

        for model in self.models:

            avg_error = 0

            validate_num = math.ceil(len(model.train_Y)/10)

            model.train_Y = np.reshape(model.train_Y, (-1, 1))
            dataset = np.concatenate((model.train_X, model.train_Y), axis=1)
            np.random.shuffle(dataset)

            error = 0

            for i in range(10):

                model.train_X = np.concatenate((dataset[(i+1)*validate_num:, :-1], dataset[:i*validate_num, :-1]), axis=0)
                model.train_Y = np.concatenate((dataset[(i+1)*validate_num:, -1], dataset[:i*validate_num, -1]), axis=0)
                model.init_W()
                model.train()
                validate_X = dataset[i*validate_num:(i+1)*validate_num, :-1]
                validate_Y = dataset[i*validate_num:(i+1)*validate_num, -1]

                if hasattr(model, 'class_list'):
                    error = error + model.calculate_avg_error_all_class(validate_X, validate_Y, model.W)
                else:
                    error = error + model.calculate_avg_error(validate_X, validate_Y, model.W)

            model.train_X = dataset[:, :-1]
            model.train_Y = dataset[:, -1]

            dataset = None
            avg_error = error/10
            self.avg_errors.append(avg_error)

        return self.avg_errors

    def get_min_avg_error(self):

        return min(self.avg_errors)

    def get_best_model(self):

        min_error_index = self.avg_errors.index(min(self.avg_errors))
        return self.models[min_error_index]


class Serializer(object):

    @staticmethod
    def serialize(need_serialize=''):
        """serialize"""
        return pickle.dumps(need_serialize, protocol=0)

    @staticmethod
    def deserialize(pickle_serialized=''):
        """ddserialize"""
        return pickle.loads(pickle_serialized)
