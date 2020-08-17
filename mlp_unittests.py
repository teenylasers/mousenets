import unittest
import numpy as np
from constants import *
from mlp import *


class MLPTest(unittest.TestCase):

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
        w = np.arange(6).reshape(2,3) * 0.1
        mlp.add_layer(n1, 'sigmoid', w)
        mlp.define_loss_function('cce')
        x0 = mlp.normalize_data(np.array([1.5, 2.5]))
        y = np.array([1, 0])

        # Check sigmoid
        s = mlp.layers[0].forward_pass(np.append(x0,1), save=False)
        assert(self._check_sigmoid(x0, mlp.get_layer(1).get_weights(), s))

        # Run forward pass once, check gradient
        x_out = mlp.forward_pass(x0)
        assert(self._check_softmax(x0, mlp.get_layer(1).get_weights(), x_out))
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
        w = np.arange(6).reshape(2,3) * 0.1
        mlp.add_layer(n1, 'softmax', w)
        mlp.define_loss_function('cce')
        x0 = mlp.normalize_data(np.array([1.5, 2.5]))
        y = np.array([1, 0])

        # Check sigmoid
        s = mlp.layers[0].forward_pass(np.append(x0,1), save=False)
        assert(self._check_softmax(x0, mlp.get_layer(1).get_weights(), s))

        # Run forward pass once, check gradient
        x_out = mlp.forward_pass(x0)
        assert(self._check_softmax(x0, mlp.get_layer(1).get_weights(), x_out))
        loss = mlp.evaluate_loss(x_out, y)
        loss_grad = mlp.calculate_loss_gradient(x_out, y)

        # Check gradient
        assert(mlp.check_gradient_from_layer(1, y, loss_grad))


if __name__ == '__main__':
    unittest.main()
