# encoding=utf8

from test_base import BaseTestCase
import FukuML.AdaBoostDecisionTree as adaboost_decision_tree


class AdaBoostDecisionTreeClassifierTestCase(BaseTestCase):

    def test_adaboost_decision_tree_classifier(self):

        #------------------------------------------------------------

        adaboost_decision_tree_c = adaboost_decision_tree.Classifier()
        adaboost_decision_tree_c.load_train_data()
        adaboost_decision_tree_c.load_test_data()
        adaboost_decision_tree_c.set_param(run_t=10)
        adaboost_decision_tree_c.init_W()
        adaboost_decision_tree_c.train()

        test_data = '-9.706 1.392 6.562 -6.543 -1.980 -6.261 -6.067 1.254 -1.071 1'
        prediction = adaboost_decision_tree_c.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("平均錯誤率（Ein）：")
        print(adaboost_decision_tree_c.calculate_avg_error(adaboost_decision_tree_c.train_X, adaboost_decision_tree_c.train_Y, adaboost_decision_tree_c.W))
        print("平均錯誤率（Eout）：")
        print(adaboost_decision_tree_c.calculate_avg_error(adaboost_decision_tree_c.test_X, adaboost_decision_tree_c.test_Y, adaboost_decision_tree_c.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
