#encoding=utf8

import os
import itertools
import numpy as np
import cvxopt
import cvxopt.solvers
import FukuML.Utility as utility
import FukuML.MLBase as ml
import FukuML.LinearRegression as linear_regression


class BinaryClassifier(ml.Learner):

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

        self.svm_kernel = 'primal_hard_margin'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.alpha = []
        self.sv_index = []
        self.sv_alpha = []
        self.sv_X = []
        self.sv_Y = []
        self.free_sv_index = []
        self.free_sv_alpha = []
        self.free_sv_X = []
        self.free_sv_Y = []
        self.sv_avg_b = 0

    def load_train_data(self, input_data_file=''):

        '''
        Load train data
        Please check dataset/pla_binary_train.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pla_binary_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file)

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        '''
        Load test data
        Please check dataset/pocket_pla_binary_test.dat to understand the data format
        Each feature of data x separated with spaces
        And the ground truth y put in the end of line separated by a space
        '''

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/pla_binary_test.dat"))
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

    def set_param(self, svm_kernel='primal_hard_margin', zeta=0, gamma=1, Q=1, C=0.1):

        # larger C => weaker regularization, smaller C => stronger regularization
        self.svm_kernel = svm_kernel
        self.zeta = zeta
        self.gamma = gamma
        self.Q = Q
        self.C = C

        return self.svm_kernel

    def init_W(self, mode='normal'):

        '''
        Init the W
        Simple way is init W all zeros
        '''

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.W = np.zeros(self.data_demension)

        if (self.svm_kernel != 'polynomial_kernel' and self.svm_kernel != 'gaussian_kernel'):
            if mode == 'linear_regression_accelerator':
                accelerator = linear_regression.Accelerator()
                self.W = accelerator.init_W(self)

        return self.W

    def score_function(self, x, W):
        # need refector

        '''
        Score function to calculate score
        '''

        if (self.svm_kernel == 'polynomial_kernel' or self.svm_kernel == 'gaussian_kernel' or self.svm_kernel == 'soft_polynomial_kernel' or self.svm_kernel == 'soft_gaussian_kernel'):
            x = x[1:]
            '''
            original_X = self.train_X[:, 1:]
            score = 0
            for i in range(len(self.sv_alpha)):
                if (self.svm_kernel == 'polynomial_kernel' or self.svm_kernel == 'soft_polynomial_kernel'):
                    score += self.sv_alpha[i] * self.sv_Y[i] * utility.Kernel.polynomial_kernel(self, original_X[self.sv_index[i]], x)
                elif (self.svm_kernel == 'gaussian_kernel' or self.svm_kernel == 'soft_gaussian_kernel'):
                    score += self.sv_alpha[i] * self.sv_Y[i] * utility.Kernel.gaussian_kernel(self, original_X[self.sv_index[i]], x)
            score = np.sign(score + self.sv_avg_b)
            '''
            score = np.sign(np.sum(self.sv_alpha * self.sv_Y * utility.Kernel.kernel_matrix_xX(self, x, self.sv_X)) + self.sv_avg_b)
        else:
            score = np.sign(np.inner(x, W))

        return score

    def error_function(self, y_prediction, y_truth):
        # need refector

        '''
        Error function to calculate error
        '''

        if y_prediction != y_truth:
            return 1
        else:
            return 0

    def calculate_avg_error(self, X, Y, W):

        return super(BinaryClassifier, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(BinaryClassifier, self).calculate_test_data_avg_error()

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.status = 'train'

        # P = Q, q = p, G = -A, h = -c

        if (self.svm_kernel == 'soft_polynomial_kernel' or self.svm_kernel == 'soft_gaussian_kernel'):

            original_X = self.train_X[:, 1:]

            K = utility.Kernel.kernel_matrix(self, original_X)

            P = cvxopt.matrix(np.outer(self.train_Y, self.train_Y) * K)
            q = cvxopt.matrix(np.ones(self.data_num) * -1)
            constrain1 = np.diag(np.ones(self.data_num) * -1)
            constrain2 = np.identity(self.data_num)
            G = cvxopt.matrix(np.vstack((constrain1, constrain2)))
            constrain1 = np.zeros(self.data_num) * -1
            constrain2 = np.ones(self.data_num) * self.C
            h = cvxopt.matrix(np.hstack((constrain1, constrain2)))
            A = cvxopt.matrix(self.train_Y, (1, self.data_num))
            b = cvxopt.matrix(0.0)
            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

            # Lagrange multipliers
            a = np.ravel(solution['x'])
            self.alpha = a
            # Support vectors have non zero lagrange multipliers
            sv = a > 1e-5
            self.sv_index = np.arange(len(a))[sv]
            self.sv_alpha = a[sv]
            self.sv_X = original_X[sv]
            self.sv_Y = self.train_Y[sv]

            free_sv = np.logical_and(a > 1e-5, a < self.C)
            self.free_sv_index = np.arange(len(a))[free_sv]
            self.free_sv_alpha = a[free_sv]
            self.free_sv_X = original_X[free_sv]
            self.free_sv_Y = self.train_Y[free_sv]

            '''
            sum_short_b = 0
            for i in range(len(self.free_sv_alpha)):
                sum_short_b += self.free_sv_Y[i]
                for j in range(len(self.free_sv_alpha)):
                    if (self.svm_kernel == 'soft_polynomial_kernel'):
                        sum_short_b -= self.free_sv_alpha[j] * self.free_sv_Y[j] * utility.Kernel.polynomial_kernel(self, original_X[self.free_sv_index[j]], original_X[self.free_sv_index[i]])
                    elif (self.svm_kernel == 'soft_gaussian_kernel'):
                        sum_short_b -= self.free_sv_alpha[j] * self.free_sv_Y[j] * utility.Kernel.gaussian_kernel(self, original_X[self.free_sv_index[j]], original_X[self.free_sv_index[i]])
            short_b = sum_short_b / len(self.free_sv_alpha)
            '''

            short_b = (np.sum(self.free_sv_Y) - np.sum(np.ravel(self.free_sv_alpha * self.free_sv_Y * utility.Kernel.kernel_matrix(self, self.free_sv_X)))) / len(self.free_sv_alpha)

            self.sv_avg_b = short_b

        elif (self.svm_kernel == 'polynomial_kernel' or self.svm_kernel == 'gaussian_kernel'):

            original_X = self.train_X[:, 1:]

            K = utility.Kernel.kernel_matrix(self, original_X)

            P = cvxopt.matrix(np.outer(self.train_Y, self.train_Y) * K)
            q = cvxopt.matrix(np.ones(self.data_num) * -1)
            G = cvxopt.matrix(np.diag(np.ones(self.data_num) * -1))
            h = cvxopt.matrix(np.zeros(self.data_num) * -1)
            A = cvxopt.matrix(self.train_Y, (1, self.data_num))
            b = cvxopt.matrix(0.0)
            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

            # Lagrange multipliers
            a = np.ravel(solution['x'])
            self.alpha = a
            # Support vectors have non zero lagrange multipliers
            sv = a > 1e-5
            self.sv_index = np.arange(len(a))[sv]
            self.sv_alpha = a[sv]
            self.sv_X = original_X[sv]
            self.sv_Y = self.train_Y[sv]

            '''
            sum_short_b = 0
            for i in range(len(self.sv_alpha)):
                sum_short_b += self.sv_Y[i]
                for j in range(len(self.sv_alpha)):
                    if (self.svm_kernel == 'polynomial_kernel'):
                        sum_short_b -= self.sv_alpha[j] * self.sv_Y[j] * utility.Kernel.polynomial_kernel(self, original_X[self.sv_index[j]], original_X[self.sv_index[i]])
                    elif (self.svm_kernel == 'gaussian_kernel'):
                        sum_short_b -= self.sv_alpha[j] * self.sv_Y[j] * utility.Kernel.gaussian_kernel(self, original_X[self.sv_index[j]], original_X[self.sv_index[i]])
            short_b = sum_short_b / len(self.sv_alpha)
            '''

            short_b = (np.sum(self.sv_Y) - np.sum(np.ravel(self.sv_alpha * self.sv_Y * utility.Kernel.kernel_matrix(self, self.sv_X)))) / len(self.sv_alpha)

            self.sv_avg_b = short_b

        elif (self.svm_kernel == 'dual_hard_margin'):

            original_X = self.train_X[:, 1:]

            P = cvxopt.matrix(np.outer(self.train_Y, self.train_Y) * np.dot(original_X, np.transpose(original_X)))
            q = cvxopt.matrix(np.ones(self.data_num) * -1)
            G = cvxopt.matrix(np.diag(np.ones(self.data_num) * -1))
            h = cvxopt.matrix(np.zeros(self.data_num) * -1)
            A = cvxopt.matrix(self.train_Y, (1, self.data_num))
            b = cvxopt.matrix(0.0)
            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

            # Lagrange multipliers
            a = np.ravel(solution['x'])
            self.alpha = a
            # Support vectors have non zero lagrange multipliers
            sv = a > 1e-5
            self.sv_index = np.arange(len(a))[sv]
            self.sv_alpha = a[sv]
            self.sv_X = original_X[sv]
            self.sv_Y = self.train_Y[sv]

            short_w = np.zeros(self.data_demension-1)
            for i in range(len(self.sv_alpha)):
                short_w += self.sv_alpha[i] * self.sv_Y[i] * self.sv_X[i]

            sum_short_b = 0
            for i in range(len(self.sv_alpha)):
                sum_short_b += self.sv_Y[i] - np.dot(np.transpose(short_w), original_X[self.sv_index[i]])
            short_b = sum_short_b / len(self.sv_alpha)

            self.sv_avg_b = short_b

            self.W = np.insert(short_w, 0, short_b)

        else:
            # primal_hard_margin
            eye_process = np.eye(self.data_demension)
            eye_process[0][0] = 0
            P = cvxopt.matrix(eye_process)
            q = cvxopt.matrix(np.zeros(self.data_demension))
            G = cvxopt.matrix(np.reshape(self.train_Y, (-1, 1)) * self.train_X * -1)
            h = cvxopt.matrix(np.ones(self.data_num) * -1)
            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h)
            self.W = np.array(solution['x'])
            self.W = np.ravel(self.W)

        return self.W

    def get_marge(self):

        nonzero = np.count_nonzero(self.W[1:])

        if nonzero == 0:
            return 0
        else:
            return 1/np.linalg.norm(self.W[1:])

    def get_support_vectors(self):

        return self.sv_X

    def prediction(self, input_data='', mode='test_data'):

        return super(BinaryClassifier, self).prediction(input_data, mode)


class MultiClassifier(BinaryClassifier):

    def __init__(self):

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

        self.svm_kernel = 'primal_hard_margin'
        self.zeta = 0
        self.gamma = 1
        self.Q = 1
        self.C = 0.1
        self.sv_index = []
        self.sv_alpha = []
        self.sv_X = []
        self.sv_Y = []
        self.free_sv_index = []
        self.free_sv_alpha = []
        self.free_sv_X = []
        self.free_sv_Y = []
        self.sv_avg_b = 0

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

    def set_param(self, svm_kernel='primal_hard_margin', zeta=0, gamma=1, Q=1, C=0.1):

        return super(MultiClassifier, self).set_param(svm_kernel, zeta, gamma, Q, C)

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

            svm_bc = BinaryClassifier()
            svm_bc.status = 'load_train_data'
            svm_bc.train_X = modify_X
            svm_bc.train_Y = modify_Y
            svm_bc.set_param(self.svm_kernel, self.zeta, self.gamma, self.Q, self.C)
            svm_bc.init_W()
            svm_bc.train()
            self.classifier_list[class_item] = svm_bc
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
