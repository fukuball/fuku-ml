# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.DecisionStump as decision_stump
import FukuML.DecisionTree as decision_tree


class DecisionStumpBinaryClassifierTestCase(BaseTestCase):

    def test_decision_stump_binary_classifier(self):

        #------------------------------------------------------------

        decision_stump_bc = decision_stump.BinaryClassifier()
        decision_stump_bc.load_train_data()
        decision_stump_bc.set_param()
        decision_stump_bc.init_W()
        decision_stump_bc.train()
        decision_stump_bc.load_test_data()

        print("\n訓練得出 sign：")
        print(decision_stump_bc.sign)
        print("訓練得出 feature index：")
        print(decision_stump_bc.feature_index)
        print("訓練得出 theta")
        print(decision_stump_bc.theta)
        print("平均錯誤率（Ein）：")
        print(decision_stump_bc.calculate_avg_error(decision_stump_bc.train_X, decision_stump_bc.train_Y, decision_stump_bc.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_stump_bc.calculate_avg_error(decision_stump_bc.test_X, decision_stump_bc.test_Y, decision_stump_bc.W))
        print('-'*70)

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/decision_stump_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/decision_stump_test.dat')

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data(input_train_data_file)
        decision_tree_c.set_param(learn_type='classifier', tree_height_limit=1)
        decision_tree_c.load_test_data(input_test_data_file)
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
