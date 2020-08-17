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


    def test_with_sigmoid(self):
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


    def test_with_softmax(self):
        # Set up a 2-element dense-layer with sigmoid activation function
        nx = 3
        n = 2
        x = np.array([1.5, 2.5])
        w = np.arange(6).reshape(2,3) * 0.1
        dense_layer = DenseLayer(nx, n, "softmax", w)
        y = np.array([1, 0])

        # Run forward pass to check softmax function
        s = dense_layer.forward_pass(np.append(x,1), save=True)
        assert(self._check_softmax(x, w, s))

        # Check gradient
        dLdy = y - s
        assert(dense_layer.check_gradient(dLdy))



class ConvLayerTest(unittest.TestCase):

    def _generate_testcase(self, nx, ny, nc, k, c, s, p):

        # Initiate a ConvLayer
        layer = ConvLayer(nx, ny, nc, k, c, s, p)

        # Generate a random input matrix
        x = np.random.rand(nc, nx, ny)

        # Generate a kernel and bias for deterministic gradient calculation
        w = np.random.rand(c, nc, k, k)
        b = np.random.rand(c, 1)

        # Generate a random expected output
        nxo = (nx + 2*p - k)/s + 1
        nyo = (ny + 2*p - k)/s + 1
        y = np.random.rand(c, int(nxo), int(nyo))

        # Return the test case
        return (layer, x, w, b, y)


    def test_conv_layer(self):
        nx, ny, nc = 8, 8, 3
        k, c = 2, 2
        s, p = 1, 1
        (layer, x, w, b, y) = self._generate_testcase(nx, ny, nc, k, c, s, p)
        output = layer.forward_pass(x, save=True, w=w, b=b)
        dLdy = y - output
        assert(layer.check_gradient(dLdy))



if __name__ == '__main__':
    unittest.main()
