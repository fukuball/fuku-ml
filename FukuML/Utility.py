#encoding=utf8

from __future__ import division
import numpy as np
import pickle
import math
import itertools
from random import randint
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import legendre
from scipy.spatial.distance import pdist, cdist, squareform


class DatasetLoader(object):

    @staticmethod
    def load(input_data_file='', data_type='float'):
        """load file"""
        X = []
        Y = []

        if data_type == 'float':
            with open(input_data_file) as f:
                for line in f:
                    data = line.split()
                    x = [1] + [float(v) for v in data[:-1]]
                    X.append(x)
                    Y.append(float(data[-1]))
        else:
            with open(input_data_file) as f:
                for line in f:
                    data = line.split()
                    x = [1] + [v for v in data[:-1]]
                    X.append(x)
                    Y.append(data[-1])

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

    @staticmethod
    def modify_XY(X, Y, class_item):

        modify_X = []
        modify_Y = []

        for idx, val in enumerate(Y):
            if val == class_item[0]:
                modify_Y.append(float(1))
                modify_X.append(X[idx])
            elif val == class_item[1]:
                modify_Y.append(float(-1))
                modify_X.append(X[idx])

        return np.array(modify_X), np.array(modify_Y)

    @staticmethod
    def modify_Y(Y, class_item):

        modify_Y = []
        for yi in Y:
            if yi == class_item:
                modify_Y.append(float(1))
            else:
                modify_Y.append(float(-1))

        return np.array(modify_Y)

    @staticmethod
    def bootstrap_bagging(X, Y, data_num):

        bootstrap_bagging_X = []
        bootstrap_bagging_Y = []

        max_row_index = data_num-1

        for _ in itertools.repeat(None, data_num):
            rand_row_index = randint(0, max_row_index)
            bootstrap_bagging_X.append(X[rand_row_index])
            bootstrap_bagging_Y.append(Y[rand_row_index])

        return np.array(bootstrap_bagging_X), np.array(bootstrap_bagging_Y)


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

            validate_num = int(math.ceil(len(model.train_Y)/10))

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


class Kernel(object):

    @staticmethod
    def kernel_matrix(svm_model, original_X):

        if (svm_model.svm_kernel == 'polynomial_kernel' or svm_model.svm_kernel == 'soft_polynomial_kernel'):
            K = (svm_model.zeta + svm_model.gamma * np.dot(original_X, original_X.T)) ** svm_model.Q
        elif (svm_model.svm_kernel == 'gaussian_kernel' or svm_model.svm_kernel == 'soft_gaussian_kernel'):
            pairwise_dists = squareform(pdist(original_X, 'euclidean'))
            K = np.exp(-svm_model.gamma * (pairwise_dists ** 2))

        '''
        K = np.zeros((svm_model.data_num, svm_model.data_num))

        for i in range(svm_model.data_num):
            for j in range(svm_model.data_num):
                if (svm_model.svm_kernel == 'polynomial_kernel' or svm_model.svm_kernel == 'soft_polynomial_kernel'):
                    K[i, j] = Kernel.polynomial_kernel(svm_model, original_X[i], original_X[j])
                elif (svm_model.svm_kernel == 'gaussian_kernel' or svm_model.svm_kernel == 'soft_gaussian_kernel'):
                    K[i, j] = Kernel.gaussian_kernel(svm_model, original_X[i], original_X[j])
        '''

        return K

    @staticmethod
    def kernel_matrix_xX(svm_model, original_x, original_X):

        if (svm_model.svm_kernel == 'polynomial_kernel' or svm_model.svm_kernel == 'soft_polynomial_kernel'):
            K = (svm_model.zeta + svm_model.gamma * np.dot(original_x, original_X.T)) ** svm_model.Q
        elif (svm_model.svm_kernel == 'gaussian_kernel' or svm_model.svm_kernel == 'soft_gaussian_kernel'):
            K = np.exp(-svm_model.gamma * (cdist(original_X, np.atleast_2d(original_x), 'euclidean').T ** 2)).ravel()

        '''
        K = np.zeros((svm_model.data_num, svm_model.data_num))

        for i in range(svm_model.data_num):
            for j in range(svm_model.data_num):
                if (svm_model.svm_kernel == 'polynomial_kernel' or svm_model.svm_kernel == 'soft_polynomial_kernel'):
                    K[i, j] = Kernel.polynomial_kernel(svm_model, original_x, original_X[j])
                elif (svm_model.svm_kernel == 'gaussian_kernel' or svm_model.svm_kernel == 'soft_gaussian_kernel'):
                    K[i, j] = Kernel.gaussian_kernel(svm_model, original_x, original_X[j])
        '''

        return K

    @staticmethod
    def polynomial_kernel(svm_model, x1, x2):

        return (svm_model.zeta + svm_model.gamma * np.dot(x1, x2)) ** svm_model.Q

    @staticmethod
    def gaussian_kernel(svm_model, x1, x2):

        return np.exp(-svm_model.gamma * (np.linalg.norm(x1-x2) ** 2))


class Serializer(object):

    @staticmethod
    def serialize(need_serialize=''):
        """serialize"""
        return pickle.dumps(need_serialize, protocol=0)

    @staticmethod
    def deserialize(pickle_serialized=''):
        """ddserialize"""
        return pickle.loads(pickle_serialized)
