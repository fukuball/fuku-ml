# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.DecisionTree as decision_tree


class DecisionTreeTestCase(BaseTestCase):

    def test_decision_tree(self):

        #------------------------------------------------------------

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data()
        decision_tree_c.set_param(learn_type='classifier')
        decision_tree_c.load_test_data()
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        test_data = '6.0 2.2 5.0 1.5 virginica'
        prediction = decision_tree_c.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None None None 1.5'
        prediction = decision_tree_c.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data()
        decision_tree_c.set_param(learn_type='classifier', prune_gain=0.1, prune_notify=True)
        decision_tree_c.load_test_data()
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        test_data = '6.0 2.2 5.0 1.5 virginica'
        prediction = decision_tree_c.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None None None 1.5'
        prediction = decision_tree_c.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/decision_tree_2_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/decision_tree_2_test.dat')

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data(input_train_data_file)
        decision_tree_c.set_param(learn_type='classifier')
        decision_tree_c.load_test_data(input_test_data_file)
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        test_data = 'ohne leicht Streifen normal normal Tuberkulose'
        prediction = decision_tree_c.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None leicht None Flocken fiepend'
        prediction = decision_tree_c.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/linear_regression_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/linear_regression_test.dat')

        decision_tree_r = decision_tree.CART()
        decision_tree_r.load_train_data(input_train_data_file)
        decision_tree_r.set_param(learn_type='regression')
        decision_tree_r.load_test_data(input_test_data_file)
        decision_tree_r.init_W()
        decision_tree_r.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_r.plot(decision_tree_r.decision_tree)

        test_data = '55.7 43 285'
        prediction = decision_tree_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None 43'
        prediction = decision_tree_r.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤值（Ein）：")
        print(decision_tree_r.calculate_avg_error(decision_tree_r.train_X, decision_tree_r.train_Y, decision_tree_r.W))
        print("W 平均錯誤值（Eout）：")
        print(decision_tree_r.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/linear_regression_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/linear_regression_test.dat')

        decision_tree_r = decision_tree.CART()
        decision_tree_r.load_train_data(input_train_data_file)
        decision_tree_r.set_param(learn_type='regression', prune_gain=0.3, prune_notify=True)
        decision_tree_r.load_test_data(input_test_data_file)
        decision_tree_r.init_W()
        decision_tree_r.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_r.plot(decision_tree_r.decision_tree)

        test_data = '55.7 43 285'
        prediction = decision_tree_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None 43'
        prediction = decision_tree_r.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤值（Ein）：")
        print(decision_tree_r.calculate_avg_error(decision_tree_r.train_X, decision_tree_r.train_Y, decision_tree_r.W))
        print("W 平均錯誤值（Eout）：")
        print(decision_tree_r.calculate_test_data_avg_error())
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
