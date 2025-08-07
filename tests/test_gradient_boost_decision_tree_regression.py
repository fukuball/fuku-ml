# encoding=utf8

from test_base import BaseTestCase
import FukuML.GradientBoostDecisionTree as gradient_boost_decision_tree


class GradientBoostDecisionTreeRegressionTestCase(BaseTestCase):

    def test_gradient_boost_decision_tree_regression(self):

        #------------------------------------------------------------

        gradient_boost_decision_tree_r = gradient_boost_decision_tree.Regression()
        gradient_boost_decision_tree_r.load_train_data()
        gradient_boost_decision_tree_r.load_test_data()
        gradient_boost_decision_tree_r.set_param(run_t=10)
        gradient_boost_decision_tree_r.init_W()
        gradient_boost_decision_tree_r.train()

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = gradient_boost_decision_tree_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("平均錯誤值（Ein）：")
        print(gradient_boost_decision_tree_r.calculate_avg_error(gradient_boost_decision_tree_r.train_X, gradient_boost_decision_tree_r.train_Y, gradient_boost_decision_tree_r.W))
        print("平均錯誤值（Eout）：")
        print(gradient_boost_decision_tree_r.calculate_avg_error(gradient_boost_decision_tree_r.test_X, gradient_boost_decision_tree_r.test_Y, gradient_boost_decision_tree_r.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
