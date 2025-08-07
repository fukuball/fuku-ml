# encoding=utf8

from test_base import BaseTestCase
import FukuML.NeuralNetwork as nn


class NeuralNetworkTestCase(BaseTestCase):

    def test_neural_network(self):

        neural_network = nn.NeuralNetwork()
        neural_network.load_train_data()
        neural_network.load_test_data()
        neural_network.set_param(network_structure=[8, 3, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000)
        neural_network.init_W()
        neural_network.train()

        print("\n訓練得出權重模型：")
        print(neural_network.W)

        print("W 平均錯誤值（Ein）：")
        print(neural_network.calculate_avg_error(neural_network.train_X, neural_network.train_Y, neural_network.W))
        print("W 平均錯誤值（Eout）：")
        print(neural_network.calculate_avg_error(neural_network.test_X, neural_network.test_Y, neural_network.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
