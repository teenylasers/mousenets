import random, unittest, timeit
import tensorflow as tf
import numpy as np
from utils import *
from convolution_play import *


class TestConvolution(unittest.TestCase):
    """
    Test and time the convolution implementations.
    """

    def _generate_test_case(self, p=False, s=False):
        """Generate a convolution test case, use tf.nn.convolution to compute the
        expected result."""
        # Generate input image x
        ns = 4
        nc = 3
        nx = 16
        ny = 16
        x = tf.random.uniform([ns, nx, ny, nc])

        # Generate convolution kernel
        kx = 3
        ky = 3
        c = 2
        kernel = tf.random.uniform([kx, ky, nc, c])

        # Padding
        if p:
            p = random.choice([1,2,3])
            xp = zero_padding(x, p)
        else:
            p = 0
            xp = x

        # Stride
        if s:
            s = random.choice([2,3])
        else:
            s = None

        # Convolution output
        y = tf.nn.convolution(input=xp, filters=kernel, strides=s, padding='VALID')

        return x, kernel, s, p, y


    def test_convolution_naive(self):
        """Test _convolution_naive()."""

        # First no zero padding and stride is 1
        x, kernel, s, p, y = self._generate_test_case()
        y_test = convolution_naive(x, kernel, s, p)
        assert(compare_matrices(y, y_test))

        # Add zero padding and stride
        x, kernel, s, p, y = self._generate_test_case(p=True, s=True)
        y_test = convolution_naive(x, kernel, s, p)
        assert(compare_matrices(y, y_test))



if __name__ == '__main__':
    unittest.main()
