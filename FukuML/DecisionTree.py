#encoding=utf8

import os
import collections
import numpy as np
import FukuML.Utility as utility
import FukuML.MLBase as ml


class DecisionTree:

    """Binary tree implementation with true and false branch. """
    def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, each_class_counts=None, height_position=0, is_leaf=True):
        self.col = col
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.each_class_counts = each_class_counts
        self.height_position = height_position
        self.is_leaf = is_leaf


class CART(ml.Learner):

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

        self.learn_type = 'classifier'
        self.decision_tree = None
        self.prune_gain = 0
        self.prune_notify = False
        self.tree_height_limit = 0

    def load_train_data(self, input_data_file=''):

        self.status = 'load_train_data'

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/decision_tree_1_train.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.train_X, self.train_Y

        self.train_X, self.train_Y = utility.DatasetLoader.load(input_data_file, data_type='string')

        return self.train_X, self.train_Y

    def load_test_data(self, input_data_file=''):

        if (input_data_file == ''):
            input_data_file = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), "dataset/decision_tree_1_test.dat"))
        else:
            if (os.path.isfile(input_data_file) is not True):
                print("Please make sure input_data_file path is correct.")
                return self.test_X, self.test_Y

        self.test_X, self.test_Y = utility.DatasetLoader.load(input_data_file, data_type='string')

        return self.test_X, self.test_Y

    def set_param(self, learn_type='classifier', prune_gain=0, prune_notify=False, tree_height_limit=0):

        self.learn_type = learn_type
        self.prune_gain = prune_gain
        self.prune_notify = prune_notify
        self.tree_height_limit = tree_height_limit

        return self.learn_type

    def init_W(self, mode='normal'):

        if (self.status != 'load_train_data') and (self.status != 'train'):
            print("Please load train data first.")
            return self.W

        self.status = 'init'

        self.data_num = len(self.train_Y)
        self.data_demension = len(self.train_X[0])
        self.W = np.zeros(self.data_demension)

        return self.W

    def score_function(self, x, W, with_missing_data=False):

        if self.learn_type == 'classifier':
            if with_missing_data:
                each_class_counts = self.classify_with_missing_data(x, self.decision_tree)
            else:
                each_class_counts = self.classify_without_missing_data(x, self.decision_tree)
        elif self.learn_type == 'regression':
            if with_missing_data:
                each_class_counts = self.regression_with_missing_data(x, self.decision_tree)
            else:
                each_class_counts = self.regression_without_missing_data(x, self.decision_tree)

        return max(each_class_counts, key=lambda k: each_class_counts[k])

    def error_function(self, y_prediction, y_truth):

        if self.learn_type == 'classifier':
            if y_prediction != y_truth:
                return 1
            else:
                return 0
        elif self.learn_type == 'regression':
            return (float(y_prediction) - float(y_truth)) ** 2

    def calculate_avg_error(self, X, Y, W):

        return super(CART, self).calculate_avg_error(X, Y, W)

    def calculate_test_data_avg_error(self):

        return super(CART, self).calculate_test_data_avg_error()

    def prune_by_height(self, tree, tree_height_limit):

        if not tree.true_branch.is_leaf:
            self.prune_by_height(tree.true_branch, tree_height_limit)
        if not tree.false_branch.is_leaf:
            self.prune_by_height(tree.false_branch, tree_height_limit)

        if tree.height_position > tree_height_limit:

            true_branch, false_branch = [], []

            for v, c in tree.true_branch.each_class_counts.items():
                true_branch += [v] * c
            for v, c in tree.false_branch.each_class_counts.items():
                false_branch += [v] * c

            true_false_branch = np.array(true_branch+false_branch)
            true_branch = np.array(true_branch)
            false_branch = np.array(false_branch)

            tree.true_branch, tree.false_branch, tree.is_leaf = None, None, True
            if self.learn_type == 'classifier':
                tree.each_class_counts = self.each_class_counts(true_false_branch)
            elif self.learn_type == 'regression':
                if len(true_false_branch) == 0:
                    mean = 0
                data = [float(y) for y in true_false_branch]
                mean = sum(data) / len(data)
                tree.each_class_counts = {mean: 1}

        return self.decision_tree

    def prune(self, tree):

        if not tree.true_branch.is_leaf:
            self.prune(tree.true_branch)
        if not tree.false_branch.is_leaf:
            self.prune(tree.false_branch)

        if tree.true_branch.is_leaf and tree.false_branch.is_leaf:
            true_branch, false_branch = [], []
            for v, c in tree.true_branch.each_class_counts.items():
                true_branch += [v] * c
            for v, c in tree.false_branch.each_class_counts.items():
                false_branch += [v] * c
            true_false_branch = np.array(true_branch+false_branch)
            true_branch = np.array(true_branch)
            false_branch = np.array(false_branch)

            p = float(len(true_branch)) / len(true_false_branch)
            delta = self.impurity(true_false_branch) - p*self.impurity(true_branch) - (1-p)*self.impurity(false_branch)
            if delta < self.prune_gain:
                if self.prune_notify:
                    print('A branch was pruned: gain = %f' % delta)
                tree.true_branch, tree.false_branch, tree.is_leaf = None, None, True
                if self.learn_type == 'classifier':
                    tree.each_class_counts = self.each_class_counts(true_false_branch)
                elif self.learn_type == 'regression':
                    if len(true_false_branch) == 0:
                        mean = 0
                    data = [float(y) for y in true_false_branch]
                    mean = sum(data) / len(data)
                    tree.each_class_counts = {mean: 1}

        return self.decision_tree

    def classify_with_missing_data(self, x, tree):

        if tree.is_leaf:
            # leaf
            return tree.each_class_counts
        else:
            v = x[tree.col]

            value_is_float = True
            try:
                v = float(v)
            except ValueError:
                value_is_float = False

            if v == 'None':
                true_branch = self.classify_with_missing_data(x, tree.true_branch)
                false_branch = self.classify_with_missing_data(x, tree.false_branch)
                true_branch_count = sum(true_branch.values())
                false_branch_count = sum(false_branch.values())
                true_branch_weight = float(true_branch_count)/(true_branch_count + false_branch_count)
                false_branch_weight = float(false_branch_count)/(true_branch_count + false_branch_count)
                each_class_counts = collections.defaultdict(int)
                for k, v in true_branch.items():
                    each_class_counts[k] += v*true_branch_weight
                for k, v in false_branch.items():
                    each_class_counts[k] += v*false_branch_weight
                return dict(each_class_counts)
            else:
                branch = None
                if value_is_float:
                    if v >= float(tree.value):
                        branch = tree.true_branch
                    else:
                        branch = tree.false_branch
                else:
                    if v == tree.value:
                        branch = tree.true_branch
                    else:
                        branch = tree.false_branch
            return self.classify_with_missing_data(x, branch)

    def classify_without_missing_data(self, x, tree):

        if tree.is_leaf:
            # leaf
            return tree.each_class_counts
        else:
            v = x[tree.col]

            value_is_float = True
            try:
                v = float(v)
            except ValueError:
                value_is_float = False

            branch = None

            if value_is_float:
                if v >= float(tree.value):
                    branch = tree.true_branch
                else:
                    branch = tree.false_branch
            else:
                if v == tree.value:
                    branch = tree.true_branch
                else:
                    branch = tree.false_branch

        return self.classify_without_missing_data(x, branch)

    def regression_with_missing_data(self, x, tree):

        if tree.is_leaf:
            # leaf
            return tree.each_class_counts
        else:
            v = x[tree.col]

            value_is_float = True
            try:
                v = float(v)
            except ValueError:
                value_is_float = False

            if v == 'None':
                true_branch = self.regression_with_missing_data(x, tree.true_branch)
                false_branch = self.regression_with_missing_data(x, tree.false_branch)
                true_branch_count = 0
                true_branch_sum = 0
                for key, value in list(true_branch.items()):
                    true_branch_count += value
                    true_branch_sum += key*value
                false_branch_count = 0
                false_branch_sum = 0
                for key, value in list(false_branch.items()):
                    false_branch_count += value
                    false_branch_sum += key*value

                mean = (float(true_branch_sum)+float(false_branch_sum))/(true_branch_count+false_branch_count)
                return {mean: 1}
            else:
                branch = None
                if value_is_float:
                    if v >= float(tree.value):
                        branch = tree.true_branch
                    else:
                        branch = tree.false_branch
                else:
                    if v == tree.value:
                        branch = tree.true_branch
                    else:
                        branch = tree.false_branch
            return self.regression_with_missing_data(x, branch)

    def regression_without_missing_data(self, x, tree):

        return self.classify_without_missing_data(x, tree)

    def each_class_counts(self, Y):
        each_class_counts = {}
        for y in Y:
            if y not in each_class_counts:
                each_class_counts[y] = 0
            each_class_counts[y] += 1
        return each_class_counts

    def impurity(self, Y):

        impurity = 1.0

        if self.learn_type == 'classifier':
            # calculate gini
            total_data_num = len(Y)
            each_class_counts = self.each_class_counts(Y)
            for k in each_class_counts:
                impurity -= (float(each_class_counts[k])/total_data_num)**2
        elif self.learn_type == 'regression':
            if len(Y) == 0:
                return 0
            data = [float(y) for y in Y]
            mean = sum(data) / len(data)
            variance = sum([(d-mean)**2 for d in data]) / len(data)
            impurity = variance

        return impurity

    def divide_set(self, X, Y, column, value):

        Y = np.reshape(Y, (-1, 1))
        XY = np.concatenate((X, Y), axis=1)

        splitting_function = None

        value_is_float = True

        try:
            value = float(value)
        except ValueError:
            value_is_float = False

        if value_is_float:
            # for int and float values
            splitting_function = lambda row: float(row[column]) >= value
        else:
            # for strings
            splitting_function = lambda row: row[column] == value

        list1 = [row for row in XY if splitting_function(row)]
        list2 = [row for row in XY if not splitting_function(row)]

        list1 = np.array(list1)
        list2 = np.array(list2)

        return (list1, list2)

    def grow_decision_tree_from(self, X, Y, height_position):

        if len(Y) == '0':
            return None

        impurity_score = self.impurity(Y)

        best_gain = 0.0
        best_attribute = None
        best_set = None

        for col in range(1, self.data_demension):
            column_values = X[:, col:col+1]
            for value in column_values:
                self.divide_set(X, Y, col, value[0])
                (set1, set2) = self.divide_set(X, Y, col, value[0])
                if set1.size != 0:
                    set1X = set1[:, :-1]
                    set1Y = set1[:, -1]
                else:
                    set1X = np.array([])
                    set1Y = np.array([])
                if set2.size != 0:
                    set2X = set2[:, :-1]
                    set2Y = set2[:, -1]
                else:
                    set2X = np.array([])
                    set2Y = np.array([])

                p = float(len(set1Y)) / len(Y)
                gain = impurity_score - p*self.impurity(set1Y) - (1-p)*self.impurity(set2Y)
                if gain > best_gain and len(set1Y) > 0 and len(set2Y) > 0:
                    best_gain = gain
                    best_attribute = (col, value[0])
                    best_set = (set1X, set1Y, set2X, set2Y)

        if best_gain > 0:
            true_branch = self.grow_decision_tree_from(best_set[0], best_set[1], height_position+1)
            false_branch = self.grow_decision_tree_from(best_set[2], best_set[3], height_position+1)
            is_leaf = False
            if true_branch is None and false_branch is None:
                is_leaf = True
            if self.learn_type == 'classifier':
                return DecisionTree(col=best_attribute[0], value=best_attribute[1], true_branch=true_branch, false_branch=false_branch, each_class_counts=self.each_class_counts(Y), height_position=height_position, is_leaf=is_leaf)
            elif self.learn_type == 'regression':
                data = [float(y) for y in Y]
                mean = sum(data) / len(data)
                mean_dict = {mean: 1}
                return DecisionTree(col=best_attribute[0], value=best_attribute[1], true_branch=true_branch, false_branch=false_branch, each_class_counts=mean_dict, height_position=height_position, is_leaf=is_leaf)
        else:
            if self.learn_type == 'classifier':
                return DecisionTree(each_class_counts=self.each_class_counts(Y), height_position=height_position, is_leaf=True)
            elif self.learn_type == 'regression':
                data = [float(y) for y in Y]
                mean = sum(data) / len(data)
                mean_dict = {mean: 1}
                return DecisionTree(each_class_counts=mean_dict, height_position=height_position, is_leaf=True)

    def plot(self, decision_tree):

        def toString(decision_tree, indent=''):

            if decision_tree.is_leaf:
                # leaf node
                return str(decision_tree.each_class_counts)
            else:
                value_is_float = True
                try:
                    float(decision_tree.value)
                except ValueError:
                    value_is_float = False
                if value_is_float:
                    decision = 'Height %s Column %s: x >= %s?' % (decision_tree.height_position, decision_tree.col, decision_tree.value)
                else:
                    decision = 'Height %s Column %s: x == %s?' % (decision_tree.height_position, decision_tree.col, decision_tree.value)
                true_branch = indent + 'yes -> ' + toString(decision_tree.true_branch, indent + '\t\t')
                false_branch = indent + 'no  -> ' + toString(decision_tree.false_branch, indent + '\t\t')
                return (decision + '\n' + true_branch + '\n' + false_branch)

        print(toString(decision_tree))

        return

    def train(self):

        if (self.status != 'init'):
            print("Please load train data and init W first.")
            return self.W

        self.decision_tree = self.grow_decision_tree_from(self.train_X, self.train_Y, height_position=1)

        if self.prune_gain > 0:
            self.decision_tree = self.prune(self.decision_tree)
        elif self.tree_height_limit > 0:
            self.decision_tree = self.prune_by_height(self.decision_tree, self.tree_height_limit)

        self.status = 'train'

        return self.W

    def prediction(self, input_data='', mode='test_data', with_missing_data=False):

        prediction = {}

        if (self.status != 'train'):
            print("Please load train data and init W then train the W first.")
            return prediction

        if (input_data == ''):
            print("Please input test data for prediction.")
            return prediction

        if mode == 'future_data':
            data = input_data.split()
            input_data_x = [v for v in data]
            input_data_x = np.ravel(input_data_x)
            input_data_x = np.insert(input_data_x, 0, '1')
            prediction = self.score_function(input_data_x, self.W, with_missing_data)
            return {"input_data_x": input_data_x, "input_data_y": None, "prediction": prediction}
        else:
            data = input_data.split()
            input_data_x = [v for v in data[:-1]]
            input_data_x = np.ravel(input_data_x)
            input_data_x = np.insert(input_data_x, 0, '1')
            input_data_y = data[-1]
            prediction = self.score_function(input_data_x, self.W, with_missing_data)
            return {"input_data_x": input_data_x, "input_data_y": input_data_y, "prediction": prediction}
