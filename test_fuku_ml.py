#encoding=utf8

import os
import unittest
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.LinearRegression as linear_regression
import FukuML.LogisticRegression as logistic_regression
import FukuML.L2RLogisticRegression as l2r_logistic_regression
import FukuML.RidgeRegression as ridge_regression
import FukuML.SupportVectorMachine as svm
import FukuML.DecisionStump as decision_stump
import FukuML.ProbabilisticSVM as probabilistic_svm
import FukuML.Utility as utility


class FukuMLTestCase(unittest.TestCase):

    def test_utility_method(self):

        utility.DatasetLoader.gen_lin_separable_data()
        utility.DatasetLoader.gen_non_lin_separable_data()
        utility.DatasetLoader.gen_lin_separable_overlap_data()

    def test_pla_binary_classifier(self):

        #------------------------------------------------------------

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data()
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data()
        pla_bc.set_param()
        pla_bc.init_W('linear_regression_accelerator')
        W = pla_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pla_binary_train.dat')
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        print("測試載入 Custom Dataset")
        print("訓練得出權重模型：")
        print(W)
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Random Cycle：")

        pla_bc.set_param(loop_mode='random')
        pla_bc.init_W()
        W = pla_bc.train()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Random Cycle alpha=0.5 step correction：")

        pla_bc.set_param(loop_mode='random', step_alpha=0.5)
        pla_bc.init_W()
        W = pla_bc.train()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        serialized_pla_bc = utility.Serializer.serialize(pla_bc)
        deserialized_pla_bc = utility.Serializer.deserialize(serialized_pla_bc)
        self.assertTrue((pla_bc.train_X == deserialized_pla_bc.train_X).all())

    def test_pla_multi_classifier(self):

        #------------------------------------------------------------

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.load_test_data()
        pla_mc.set_param()
        pla_mc.init_W()
        W = pla_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = pla_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.train_X, pla_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.test_X, pla_mc.test_Y, W))
        print('-'*70)

        print("使用 Linear Regression 加速器：")
        pla_mc.set_param()
        pla_mc.init_W('linear_regression_accelerator')
        W = pla_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = pla_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("W 平均錯誤率（Ein）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.train_X, pla_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.test_X, pla_mc.test_Y, W))
        print('-'*70)

    def test_pocket_pla_binary_classifier(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("Dataset 不是線性可分的，所以要使用 Pocket PLA.")
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data()
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = pocket_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        serialized_pocket_bc = utility.Serializer.serialize(pocket_bc)
        deserialized_pocket_bc = utility.Serializer.deserialize(serialized_pocket_bc)
        self.assertTrue((pocket_bc.train_X == deserialized_pocket_bc.train_X).all())

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data()
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W('linear_regression_accelerator')
        W = pocket_bc.train()
        pocket_bc.load_test_data()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

    def test_pocket_pla_multi_classifier(self):

        #------------------------------------------------------------

        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.load_test_data()
        pocket_mc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_mc.init_W()
        W = pocket_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = pocket_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.train_X, pocket_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.test_X, pocket_mc.test_Y, W))
        print('-'*70)

        print("使用 Linear Regression 加速器：")
        pocket_mc.set_param()
        pocket_mc.init_W('linear_regression_accelerator')
        W = pocket_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = pocket_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("W 平均錯誤率（Ein）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.train_X, pocket_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.test_X, pocket_mc.test_Y, W))
        print('-'*70)

    def test_linear_regression(self):

        #------------------------------------------------------------

        linear = linear_regression.LinearRegression()
        linear.load_train_data()
        linear.load_test_data()
        linear.set_param()
        linear.init_W()
        W = linear.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = linear.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(linear.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(linear.calculate_avg_error(linear.train_X, linear.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(linear.calculate_avg_error(linear.test_X, linear.test_Y, W))
        print('-'*70)

    def test_linear_regression_binary_classifier(self):

        #------------------------------------------------------------

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data()
        linear_bc.load_test_data()
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = linear_bc.prediction(test_data)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

    def test_linear_regression_multi_classifier(self):

        #------------------------------------------------------------

        linear_mc = linear_regression.MultiClassifier()
        linear_mc.load_train_data()
        linear_mc.load_test_data()
        linear_mc.set_param()
        linear_mc.init_W()
        W = linear_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = linear_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.train_X, linear_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.test_X, linear_mc.test_Y, W))
        print('-'*70)

    def test_logistic_regression(self):

        #------------------------------------------------------------

        logistic = logistic_regression.LogisticRegression()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        logistic = logistic_regression.LogisticRegression()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W('linear_regression_accelerator')
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

    def test_logistic_regression_binary_classifier(self):

        #------------------------------------------------------------

        logistic = logistic_regression.BinaryClassifier()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        print("隨機梯度下降：")
        logistic.set_param(feed_mode='stochastic', step_eta=0.1, updates=2000)
        logistic.init_W()
        W = logistic.train()
        print("訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")
        logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000)
        logistic.init_W('linear_regression_accelerator')
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

    def test_logistic_regression_multi_classifier(self):

        #------------------------------------------------------------

        logistic = logistic_regression.MultiClassifier()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = logistic.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        print("隨機梯度下降：")
        logistic.set_param(feed_mode='stochastic', step_eta=0.126, updates=2000)
        logistic.init_W()
        W = logistic.train()
        print("訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")
        logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000)
        W = logistic.init_W('linear_regression_accelerator')
        print("\n加速初始化權重模型：")
        print(W)
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("One to One Multiclassification：")
        logistic.set_param()
        logistic.init_W('normal', 'ovo')
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = logistic.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("One to One Multiclassification 使用 Linear Regression 加速器：")
        logistic.set_param()
        logistic.init_W('linear_regression_accelerator', 'ovo')
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = logistic.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

    def test_l2r_logistic_regression(self):

        #------------------------------------------------------------

        l2r_logistic = l2r_logistic_regression.L2RLogisticRegression()
        l2r_logistic.load_train_data()
        l2r_logistic.load_test_data()
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = l2r_logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(l2r_logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

    def test_l2r_logistic_regression_binary_classifier(self):

        #------------------------------------------------------------

        l2r_logistic = l2r_logistic_regression.BinaryClassifier()
        l2r_logistic.load_train_data()
        l2r_logistic.load_test_data()
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = l2r_logistic.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

        print("隨機梯度下降：")
        l2r_logistic.set_param(feed_mode='stochastic', step_eta=0.1, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()
        print("訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W('linear_regression_accelerator')
        W = l2r_logistic.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

    def test_binary_classifier_with_feature_transform(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\nPLA 訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_feature_transform('polynomial', 2)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\nPLA Polynomial 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_feature_transform('legendre', 2)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\nPLA Legendre 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data(input_test_data_file)

        print("Pocket 訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_feature_transform('polynomial', 2)
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data(input_test_data_file)

        print("Pocket Polynomial 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_feature_transform('legendre', 2)
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data(input_test_data_file)

        print("Pocket Legendre 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()
        linear_bc.load_test_data(input_test_data_file)

        print("\nLinear Regression 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_feature_transform('polynomial', 2)
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()
        linear_bc.load_test_data(input_test_data_file)

        print("\nLinear Regression Polynomial 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_feature_transform('legendre', 2)
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()
        linear_bc.load_test_data(input_test_data_file)

        print("\nLinear Regression Legendre 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_param()
        logistic_bc.init_W()
        W = logistic_bc.train()
        logistic_bc.load_test_data(input_test_data_file)

        print("\nLogistic Regression 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.train_X, logistic_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.test_X, logistic_bc.test_Y, W))
        print('-'*70)

        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_feature_transform('polynomial', 2)
        logistic_bc.set_param()
        logistic_bc.init_W()
        W = logistic_bc.train()
        logistic_bc.load_test_data(input_test_data_file)

        print("\nLogistic Regression Polynomial 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.train_X, logistic_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.test_X, logistic_bc.test_Y, W))
        print('-'*70)

        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_feature_transform('legendre', 2)
        logistic_bc.set_param()
        logistic_bc.init_W()
        W = logistic_bc.train()
        logistic_bc.load_test_data(input_test_data_file)

        print("\nLogistic Regression Legendre 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.train_X, logistic_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.test_X, logistic_bc.test_Y, W))
        print('-'*70)

    def test_multi_classifier_with_feature_transform(self):

        #------------------------------------------------------------

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_feature_transform('polynomial', 1)
        pla_mc.load_test_data()
        pla_mc.set_param()
        pla_mc.init_W()
        W = pla_mc.train()
        print("\nPLA Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.train_X, pla_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.test_X, pla_mc.test_Y, W))
        print('-'*70)

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_feature_transform('legendre', 1)
        pla_mc.load_test_data()
        pla_mc.set_param()
        pla_mc.init_W()
        W = pla_mc.train()
        print("\nPLA Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.train_X, pla_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.test_X, pla_mc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_feature_transform('polynomial', 1)
        pocket_mc.load_test_data()
        pocket_mc.set_param()
        pocket_mc.init_W()
        W = pocket_mc.train()
        print("\nPocket Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.train_X, pocket_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.test_X, pocket_mc.test_Y, W))
        print('-'*70)

        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_feature_transform('legendre', 1)
        pocket_mc.load_test_data()
        pocket_mc.set_param()
        pocket_mc.init_W()
        W = pocket_mc.train()
        print("\nPocket Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.train_X, pocket_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.test_X, pocket_mc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        linear_mc = linear_regression.MultiClassifier()
        linear_mc.load_train_data()
        linear_mc.set_feature_transform('polynomial', 1)
        linear_mc.load_test_data()
        linear_mc.set_param()
        linear_mc.init_W()
        W = linear_mc.train()
        print("\nLinear Regression Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.train_X, linear_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.test_X, linear_mc.test_Y, W))
        print('-'*70)

        linear_mc = linear_regression.MultiClassifier()
        linear_mc.load_train_data()
        linear_mc.set_feature_transform('legendre', 1)
        linear_mc.load_test_data()
        linear_mc.set_param()
        linear_mc.init_W()
        W = linear_mc.train()
        print("\nLinear Regression Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.train_X, linear_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.test_X, linear_mc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        logistic_mc = logistic_regression.MultiClassifier()
        logistic_mc.load_train_data()
        logistic_mc.set_feature_transform('polynomial', 1)
        logistic_mc.load_test_data()
        logistic_mc.set_param()
        logistic_mc.init_W()
        W = logistic_mc.train()

        print("\nLogistic Regression Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.train_X, logistic_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.test_X, logistic_mc.test_Y, W))
        print('-'*70)

        logistic_mc = logistic_regression.MultiClassifier()
        logistic_mc.load_train_data()
        logistic_mc.set_feature_transform('legendre', 1)
        logistic_mc.load_test_data()
        logistic_mc.set_param()
        logistic_mc.init_W()
        W = logistic_mc.train()

        print("\nLogistic Regression Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.train_X, logistic_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.test_X, logistic_mc.test_Y, W))
        print('-'*70)

    def test_binary_classification_cross_validation(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        cross_validator = utility.CrossValidator()

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_param()
        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_param()
        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_param()

        print("\n10 fold cross validation：")

        cross_validator.add_model(pla_bc)
        cross_validator.add_model(pocket_bc)
        cross_validator.add_model(linear_bc)
        cross_validator.add_model(logistic_bc)
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)
        best_model.init_W()
        W = best_model.train()
        best_model.load_test_data(input_test_data_file)

        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error(best_model.test_X, best_model.test_Y, W))
        print('-'*70)

    def test_multi_classification_cross_validation(self):

        cross_validator = utility.CrossValidator()

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_param()
        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_param()

        print("\n10 fold cross validation：")

        cross_validator.add_model(pla_mc)
        cross_validator.add_model(pocket_mc)
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)
        best_model.init_W()
        W = best_model.train()
        best_model.load_test_data()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error_all_class(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error_all_class(best_model.test_X, best_model.test_Y, W))
        print('-'*70)

    def test_ridge_regression(self):

        #------------------------------------------------------------

        ridge = ridge_regression.RidgeRegression()
        ridge.load_train_data()
        ridge.load_test_data()
        ridge.set_param(lambda_p=pow(10, -3))
        ridge.init_W()
        W = ridge.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = ridge.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(ridge.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(ridge.calculate_avg_error(ridge.train_X, ridge.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(ridge.calculate_avg_error(ridge.test_X, ridge.test_Y, W))
        print('-'*70)

    def test_ridge_regression_binary_classifier(self):

        #------------------------------------------------------------

        cross_validator = utility.CrossValidator()

        lambda_p_set = [i for i in range(2, -11, -1)]

        for lambda_p in lambda_p_set:
            ridge_bc = ridge_regression.BinaryClassifier()
            ridge_bc.load_train_data()
            ridge_bc.set_param(lambda_p=pow(10, lambda_p))
            cross_validator.add_model(ridge_bc)

        print("\n10 fold cross validation：")
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)

        best_model.load_test_data()
        best_model.init_W()
        W = best_model.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.402041 0.402048 -1'
        prediction = best_model.prediction(test_data)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error(best_model.test_X, best_model.test_Y, W))
        print('-'*70)

    def test_ridge_regression_multi_classifier(self):

        #------------------------------------------------------------

        ridge_mc = ridge_regression.MultiClassifier()
        ridge_mc.load_train_data()
        ridge_mc.load_test_data()
        ridge_mc.set_param(lambda_p=pow(10, -3))
        ridge_mc.init_W()
        W = ridge_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = ridge_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(ridge_mc.calculate_avg_error_all_class(ridge_mc.train_X, ridge_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(ridge_mc.calculate_avg_error_all_class(ridge_mc.test_X, ridge_mc.test_Y, W))
        print('-'*70)

    def test_primal_hard_margin_svm_binary_classifier(self):

        #------------------------------------------------------------

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data()
        svm_bc.load_test_data()
        svm_bc.set_param(svm_kernel='primal_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print('-'*70)

    def test_dual_hard_margin_svm_binary_classifier(self):

        #------------------------------------------------------------

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data()
        svm_bc.load_test_data()
        svm_bc.set_param(svm_kernel='dual_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print('-'*70)

    def test_non_linear_hard_margin_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.set_feature_transform('legendre', 3)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='primal_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())

        test_data = '-3.53133797 5.23661847 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.set_feature_transform('legendre', 3)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='dual_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_polynomial_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='polynomial_kernel', zeta=100, gamma=1, Q=3)
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_gaussian_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='gaussian_kernel', gamma=0.001)
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_soft_polynomial_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='soft_polynomial_kernel', zeta=0, gamma=1, Q=1, C=0.1)
        svm_bc.init_W()
        W = svm_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_soft_gaussian_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=1000)
        svm_bc.init_W()
        W = svm_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_probabilistic_svm(self):

        #------------------------------------------------------------

        probabilistic = probabilistic_svm.ProbabilisticSVM()
        probabilistic.load_train_data()
        probabilistic.load_test_data()
        probabilistic.set_param()
        probabilistic.init_W()
        probabilistic.train()

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = probabilistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(probabilistic.error_function(prediction['input_data_x'], prediction['input_data_y'], probabilistic.W))
        print("W 平均錯誤值（Ein）：")
        print(probabilistic.calculate_avg_error(probabilistic.train_X, probabilistic.train_Y, probabilistic.W))
        print("W 平均錯誤值（Eout）：")
        print(probabilistic.calculate_avg_error(probabilistic.test_X, probabilistic.test_Y, probabilistic.W))
        print('-'*70)

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

if __name__ == '__main__':

    unittest.main(verbosity=2)
