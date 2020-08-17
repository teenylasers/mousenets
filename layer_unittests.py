import unittest
import numpy as np
from constants import *
from layer import *


class DenseLayerTest(unittest.TestCase):


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


    def test_dense_layer_sigmoid(self):
        # Set up a 2-element dense-layer with sigmoid activation function
        nx = 3
        n = 2
        x = np.array([1.5, 2.5])
        w = np.arange(6).reshape(2,3) * 0.1
        dense_layer = DenseLayer(nx, n, "sigmoid", w)
        y = np.array([1, 0])

        # Run forward pass to check sigmoid function
        s = dense_layer.forward_pass(np.append(x,1), save=True)
        assert(self._check_sigmoid(x, w, s))

        # Check gradient
        dLdy = y - s
        assert(dense_layer.check_gradient(dLdy))



class LayerTest(unittest.TestCase):

    def test_conv_layer(self):
        assert True


if __name__ == '__main__':
    unittest.main()
