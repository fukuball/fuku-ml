#encoding=utf8

import os
import itertools
import numpy as np
import FukuML.Utility as utility
import FukuML.RidgeRegression as ridge_regression


class KernelRidgeRegression(ridge_regression.RidgeRegression):

    def __init__(self):

        """init"""

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = []
        self.data_num = 0
        self.data_demension = 0
        self.test_X = []
        self.test_Y = []
        self.feature_transform_mode = ''
        self.feature_transform_degree = 1

        self.lambda_p = 0.0001
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.beta = []

    def load_train_data(self, input_data_file=''):

        return super(KernelRidgeRegression, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(KernelRidgeRegression, self).load_test_data(input_data_file)

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        # larger C => weaker regularization, smaller C => stronger regularization
        # smaller lambda => weaker regularization, larger lambda => stronger regularization
        self.lambda_p = lambda_p
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = gamma
        self.Q = 1
        self.C = C

        return self.lambda_p, self.gamma, self.C

    def init_W(self):

        return super(KernelRidgeRegression, self).init_W()

    def score_function(self, x, W):

        x = x[1:]
        original_X = self.train_X[:, 1:]
        score = np.sum(self.beta * utility.Kernel.kernel_matrix_xX(self, x, original_X))

        return score

    def error_function(self, y_prediction, y_truth):

        return super(KernelRidgeRegression, self).error_function(y_prediction, y_truth)

    def calculate_avg_error(self, X, Y, W):

        return super(KernelRidgeRegression, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(KernelRidgeRegression, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        original_X = self.train_X[:, 1:]
        K = utility.Kernel.kernel_matrix(self, original_X)
        I = np.diag(np.ones(self.data_num))

        inverse_part = np.linalg.inv(self.lambda_p * I + K)
        self.beta = np.dot(inverse_part, self.train_Y)

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        return super(KernelRidgeRegression, self).prediction(input_data, mode)


class BinaryClassifier(KernelRidgeRegression):

    def __init__(self):

        """init"""

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = []
        self.data_num = 0
        self.data_demension = 0
        self.test_X = []
        self.test_Y = []
        self.feature_transform_mode = ''
        self.feature_transform_degree = 1

        self.lambda_p = 0.0001
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.beta = []

    def load_train_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_train_data(input_data_file)

    def load_test_data(self, input_data_file=''):

        return super(BinaryClassifier, self).load_test_data(input_data_file)

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        return super(BinaryClassifier, self).set_param(lambda_p, gamma, C)

    def init_W(self):

        return super(BinaryClassifier, self).init_W()

    def score_function(self, x, W):

        score = super(BinaryClassifier, self).score_function(x, W)
        score = np.sign(score)

        return score

    def error_function(self, y_prediction, y_truth):

        if y_prediction != y_truth:
            return 1
        else:
            return 0

    def calculate_avg_error(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        return super(BinaryClassifier, self).train()

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)


class MultiClassifier(BinaryClassifier):

    def __init__(self):

        """init"""

        self.status = 'empty'
        self.train_X = []
        self.train_Y = []
        self.W = []
        self.data_num = 0
        self.data_demension = 0
        self.test_X = []
        self.test_Y = []
        self.feature_transform_mode = ''
        self.feature_transform_degree = 1

        self.lambda_p = 0.0001
        self.svm_kernel = 'soft_gaussian_kernel'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.beta = []

        self.class_list = []
        self.classifier_list = {}
        self.decomposition = 'ovo'

    def load_train_data(self, input_data_file=''):

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/digits_multiclass_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/digits_multiclass_test.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.test_X, self.test_Y

        self.test_X, self.test_Y = utility.DatasetLoader.load(input_data_file)

        if (self.feature_transform_mode == 'polynomial') or (self.feature_transform_mode == 'legendre'):
            self.test_X = self.test_X[:, 1:]

            self.test_X = utility.DatasetLoader.feature_transform(
                self.test_X,
                self.feature_transform_mode,
                self.feature_transform_degree
            )

        return self.test_X, self.test_Y

    def set_param(self, lambda_p=0.0001, gamma=1, C=0.1):

        return super(MultiClassifier, self).set_param(lambda_p, gamma, C)

    def init_W(self, mode='normal'):

        self.W = {}

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.class_list = list(itertools.combinations(np.unique(self.train_Y), 2))

        for class_item in self.class_list:
            self.W[class_item] = np.zeros(self.data_demension)

        return self.W

    def score_function(self, x, W):

        return super(MultiClassifier, self).score_function(x, W)

    def score_function_all_class(self, x, W):

        score_list = {}
        ovo_vote = []

        for class_item in self.class_list:
            score = self.classifier_list[class_item].score_function(x, W)
            if score == 1:
                score_list[class_item] = class_item[0]
            else:
                score_list[class_item] = class_item[1]
            ovo_vote.append(score_list[class_item])

        return max(set(ovo_vote), key=ovo_vote.count)

    def error_function(self, y_prediction, y_truth):

        return super(MultiClassifier, self).error_function(y_prediction, y_truth)

    def calculate_avg_error(self, X, Y, W):

        return super(MultiClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_avg_error_all_class(self, X, Y, W):

        data_num = len(Y)
        error_num = 0

        for i in range(data_num):
            error_num = error_num + self.error_function(self.score_function_all_class(X[i], W), Y[i])

        avg_error = error_num / float(data_num)

        return avg_error

    def calculate_test_data_avg_error(self):

        return super(MultiClassifier, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        for class_item in self.class_list:

            modify_X, modify_Y = utility.DatasetLoader.modify_XY(self.train_X, self.train_Y, class_item)

            kernel_ridge_regression_bc = BinaryClassifier()
            kernel_ridge_regression_bc.status = 'load_train_data'
            kernel_ridge_regression_bc.train_X = modify_X
            kernel_ridge_regression_bc.train_Y = modify_Y
            kernel_ridge_regression_bc.set_param(self.lambda_p, self.gamma, self.C)
            kernel_ridge_regression_bc.init_W()
            kernel_ridge_regression_bc.train()
            self.classifier_list[class_item] = kernel_ridge_regression_bc
            print("class %d to %d learned." % (class_item[0], class_item[1]))

        self.status = 'train'

        return self.W

    def prediction(self, input_data='', mode='test_data'):

        prediction = {}
        prediction_list = {}
        prediction_return = 0.0
        ovo_vote = []

        for class_item in self.class_list:
            prediction = self.classifier_list[class_item].prediction(input_data, mode)
            if prediction['prediction'] == 1:
                prediction_list[class_item] = class_item[0]
            else:
                prediction_list[class_item] = class_item[1]
            ovo_vote.append(prediction_list[class_item])

        prediction_return = max(set(ovo_vote), key=ovo_vote.count)

        return {
            "input_data_x": prediction['input_data_x'],
            "input_data_y": prediction['input_data_y'],
            "prediction": prediction_return,
            "prediction_list": prediction_list,
        }
