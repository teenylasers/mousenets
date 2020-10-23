import random, unittest
import unittest_mlp, unittest_sequential_net
import numpy as np
from net_trainer import *


def generate_training_data_in(dims):
    """Generate random input training data of dimensions dims."""
    test_normalization_factor = random.uniform(1.0, 1e3)
    x_train = np.random.rand(*dims) * test_normalization_factor
    return x_train


def generate_training_data_out(dims):
    """Generate one-hot output training data of dimensions dims."""
    y_train = np.zeros(dims)
    for i in range(dims[0]):
        y_train[i][random.choice(range(dims[1]))] = 1
    return y_train



class NetTrainerTest(unittest.TestCase):
    """
    Test NetTrainer using simple MLP and SequentialNets
    """

    def test_mlp_sgd_visual(self):
        """Test training a set of defined MLP using sgd (stochastic gradient
        descent). Plot visualizations to check training process and parameter
        goodness."""
        return
        self._test_mlp_sgd(num_tests = 3, viz=True)


    def _test_mlp_sgd(self, num_tests=None, test_activations=None, viz=False):
        """Test training random MLPs using sgd (stochastic gradient descent)"""

        if num_tests is None:
            num_tests = 3

        if test_activations is None:
            test_activations = ['relu', 'sigmoid', 'softmax']

        num_samples = [32, 128]
        epochs = 10

        for activation in test_activations:
            for i in range(num_tests):

                print('MLP with %s, test #%d: ' % (activation, i))

                mlp, input_dimension, output_dimension = \
                    unittest_mlp.construct_mlp(activation)
                ns = random.choice(num_samples)
                batch_size = int(ns/4)
                x_train = generate_training_data_in((ns, input_dimension))
                y_train = generate_training_data_out((ns, output_dimension))

                trainer = NetTrainer()
                mlp, history = trainer.sgd(
                    mlp, x_train, y_train, epochs, batch_size, eta=None, viz=viz)

                print('loss history: {} -> {}'.format(
                    history.loss_history[0], history.loss_history[-1]))
                # assert(loss_history[-1] < loss_history[0]), \
                #     '{}'.format(loss_history)


    def test_convnet_sgd(self):
        """Test training random ConvNets using sgd (stochasitc gradient descent)"""

        num_tests = 3
        test_activations = ['relu', 'sigmoid']
        pooling_fxns = ['average', 'max']
        num_samples = [32, 128]
        epochs = 20
        num_output = random.choice([2,3,4,5])

        for activation in test_activations:
            for i in range(num_tests):

                print('ConvNet with %s, test #%d: ' % (activation, i))

                convnet = unittest_sequential_net.construct_convnet(
                    num_output, activation_fxn='relu', pooling_fxn='average')

                ns = random.choice(num_samples)
                batch_size = int(ns/4)
                input_dimension = (convnet.layers[1].nc,
                                   convnet.layers[1].nx,
                                   convnet.layers[1].ny)
                output_dimension = num_output

                x_train = generate_training_data_in((ns, *input_dimension))
                y_train = generate_training_data_out((ns, output_dimension))

                trainer = NetTrainer()
                convnet, history = trainer.sgd(
                    convnet, x_train, y_train, epochs, batch_size)

                print('loss history: {} -> {}'.format(
                    history.loss_history[0], history.loss_history[-1]))
                # assert(loss_history[-1] < loss_history[0]), \
                #     '{}'.format(loss_history)



if __name__ == '__main__':
    unittest.main()
