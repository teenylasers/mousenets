import random, unittest
import numpy as np
from constants import *
from mlp import *


def construct_mlp(activation_function):
    """Construct and return an MLP using MLP with layer_width and randomly
    generated weights. All layers except the last uses activation_function,
    the last layer uses softmax. Use cce loss function."""

    layer_widths = _generate_mlp_params()

    input_dimension = layer_widths[0]
    output_dimension = layer_widths[-1]
    assert output_dimension >= 2, 'output_dimension < 2 is meaningless.'
    mlp = MLP(input_dimension, output_dimension)
    for l in layer_widths[1:-1]:
        mlp.add_layer(l, activation_function)
    mlp.add_layer(layer_widths[-1], 'softmax')
    mlp.define_loss_function('cce')
    return mlp, input_dimension, output_dimension


def _generate_mlp_params():
    """Generate a random set of params for MLP."""
    layer_widths = [random.choice(range(9,36))]
    num_layers = random.choice([2,3,4])
    for l in range(num_layers):
        if layer_widths[-1] == 1:
            continue
        else:
            next_width = 2 if layer_widths[-1] == 2 \
                else random.choice(range(2,layer_widths[-1]))
            layer_widths.append(next_width)
    return layer_widths



class MLPSmallTest(unittest.TestCase):

    def _check_sigmoid(self, x, w, y):
        """Check y == sigmoid(w*x)"""
        h = w.dot(np.append(x, 1))
        x1 = 1/(1 + np.exp(-h))
        if (all([it < kAllowNumericalErr for it in (x1-y)])):
            return True
        else:
            print('Sigmoid function implementation is incorrect: \n'
                  'should be {} \ninstead got {}'.format(x1, y))
            return False

    def _check_softmax(self, x, w, y):
        """Check y == softmax(w*x)"""
        h = w.dot(np.append(x, 1))
        x1 = np.exp(h) / np.sum(np.exp(h))
        if (all([it < kAllowNumericalErr for it in (x1-y)])):
            return True
        else:
            print('Softmax function implementation is incorrect: \n'
                  'should be {} \ninstead got {}'.format(x1, y))
            return False


    def test_gradient_cce_sigmoid(self):
        """Check analytical gradient from backprop against numerically
        calculated gradient."""
        # Set up a 2-layer MLP of width 2.
        n0 = 2
        n1 = 2
        mlp = MLP(input_dimension = n0, output_dimension = n1)
        wb = np.arange(6).reshape(2,3) * 0.1
        w = wb[:,:-1]
        b = wb[:,-1:]
        mlp.add_layer(n1, 'sigmoid', w, b)
        mlp.define_loss_function('cce')
        x0 = mlp.normalize_data(np.array([1.5, 2.5]))
        y = np.array([1, 0])

        # Check sigmoid
        s = mlp.layers[0].forward_pass(x0, save=False)
        assert(self._check_sigmoid(x0, mlp.get_layer(1).wb, s))

        # Run forward pass once, check gradient
        x_out = mlp.forward_pass(x0)
        assert(self._check_softmax(x0, mlp.get_layer(1).wb, x_out))
        loss = mlp.evaluate_loss(x_out, y)
        loss_grad = mlp.calculate_loss_gradient(x_out, y)

        # Check gradient
        assert(mlp.check_gradient_from_layer(1, y, loss_grad))


    def test_gradient_cce_softmax(self):
        """Check analytical gradient from backprop against numerically
        calculated gradient."""
        # Set up a 2-layer MLP of width 2.
        n0 = 2
        n1 = 2
        mlp = MLP(input_dimension = n0, output_dimension = n1)
        wb = np.arange(6).reshape(2,3) * 0.1
        w = wb[:,:-1]
        b = wb[:,-1:]
        mlp.add_layer(n1, 'softmax', w)
        mlp.define_loss_function('cce')
        x0 = mlp.normalize_data(np.array([1.5, 2.5]))
        y = np.array([1, 0])

        # Check sigmoid
        s = mlp.layers[0].forward_pass(x0, save=False)
        assert(self._check_softmax(x0, mlp.get_layer(1).wb, s))

        # Run forward pass once, check gradient
        x_out = mlp.forward_pass(x0)
        assert(self._check_softmax(x0, mlp.get_layer(1).wb, x_out))
        loss = mlp.evaluate_loss(x_out, y)
        loss_grad = mlp.calculate_loss_gradient(x_out, y)

        # Check gradient
        assert(mlp.check_gradient_from_layer(1, y, loss_grad))



class MLPTest(unittest.TestCase):
    """
    Test a simple MLP implemented using SequentialNet.
    """

    def test_mlp_gradient_check(self):

        test_activations = ['relu', 'sigmoid', 'softmax']
        num_tests = 3

        for activation in test_activations:
            for i in range(num_tests):

                print('MLP with %s, test %d: ' % (activation, i))
                mlp, input_dimension, output_dimension = \
                    construct_mlp(activation)
                x0 = np.random.rand(input_dimension) * 234.6
                xn = mlp.forward_pass(x0)
                y = np.zeros(output_dimension)
                y[random.choice(range(output_dimension))] = 1

                loss = mlp.evaluate_loss(xn, y)
                loss_grad = mlp.calculate_loss_gradient(xn, y)

                assert(mlp.check_gradient_from_layer(1, y, loss_grad))



if __name__ == '__main__':
    unittest.main()
