import random, unittest, timeit, np_impl.utils
import tensorflow as tf
import numpy as np
from constants import *
from utils import *


class TestCompareMatrices(unittest.TestCase):
    """Test compare_matrices() function."""

    def test_compare_matrices_np(self):
        a = np.array([[1.1, 2.2],[3.3, 4.4],[5.5, 6.6]])
        b = a + np.array([[0,0],[1e-5,0],[0,1e-5]])
        assert(compare_matrices(a, b, 1e-4))
        assert(not compare_matrices(a, b, 1e-6))

    def test_compare_matrices_tf(self):
        a = tf.Variable([[1.1, 2.2],[3.3, 4.4],[5.5, 6.6]])
        b = a + tf.Variable([[0,0],[1e-5,0],[0,1e-5]])
        assert(compare_matrices(a, b, 1e-4))
        assert(not compare_matrices(a, b, 1e-6))


class TestNormalizeData(unittest.TestCase):
    """Test normalize_data() function."""

    def test_normalize_data(self):
        """Test normalize_data function for both numpy array and tf.Variable."""
        num_tests = 3
        x_dims=[2,3,4,8,16]
        y_dims=[2,3,4,8,16]
        z_dims=[2,3,4,8,16]

        # Test 2D matrices
        for ii in range(num_tests):
            x_np = np.random.rand(random.choice(x_dims), random.choice(y_dims))
            x_tf = tf.Variable(x_np)

            x_np_normalized = normalize_data(x_np)
            self.assertAlmostEqual(tf.math.reduce_mean(x_np_normalized).numpy(),
                                   0, 6,
                                   'mean = {}'.format(
                                       tf.math.reduce_mean(x_np_normalized)))
            self.assertAlmostEqual(tf.math.reduce_std(x_np_normalized).numpy(),
                                   1, 6,
                                   'stdev = {}'.format(
                                       tf.math.reduce_std(x_np_normalized)))

            x_tf_normalized = normalize_data(x_tf)
            self.assertAlmostEqual(tf.math.reduce_mean(x_tf_normalized).numpy(),
                                   0, 6,
                                   'mean = {}'.format(
                                       tf.math.reduce_mean(x_tf_normalized)))
            self.assertAlmostEqual(tf.math.reduce_std(x_tf_normalized).numpy(),
                                   1, 6,
                                   'stdev = {}'.format(
                                       tf.math.reduce_std(x_tf_normalized)))

        # Test 3D matrices
        for ii in range(num_tests):
            x_np = np.random.rand(random.choice(x_dims), random.choice(y_dims),
                                  random.choice(z_dims))
            x_tf = tf.Variable(x_np)

            x_np_normalized = normalize_data(x_np)
            self.assertAlmostEqual(tf.math.reduce_mean(x_np_normalized).numpy(),
                                   0, 6,
                                   'mean = {}'.format(
                                       tf.math.reduce_mean(x_np_normalized)))
            self.assertAlmostEqual(tf.math.reduce_std(x_np_normalized).numpy(),
                                   1, 6,
                                   'stdev = {}'.format(
                                       tf.math.reduce_std(x_np_normalized)))

            x_tf_normalized = normalize_data(x_tf)
            self.assertAlmostEqual(tf.math.reduce_mean(x_tf_normalized).numpy(),
                                   0, 6,
                                   'mean = {}'.format(
                                       tf.math.reduce_mean(x_tf_normalized)))
            self.assertAlmostEqual(tf.math.reduce_std(x_tf_normalized).numpy(),
                                   1, 6,
                                   'stdev = {}'.format(
                                       tf.math.reduce_std(x_tf_normalized)))

        # Test 1D vectors
        for ii in range(num_tests):
            x_np = np.random.rand(random.choice(x_dims))
            x_tf = tf.Variable(x_np)

            x_np_normalized = normalize_data(x_np)
            self.assertAlmostEqual(tf.math.reduce_mean(x_np_normalized).numpy(),
                                   0, 6,
                                   'mean = {}'.format(
                                       tf.math.reduce_mean(x_np_normalized)))
            self.assertAlmostEqual(tf.math.reduce_std(x_np_normalized).numpy(),
                                   1, 6,
                                   'stdev = {}'.format(
                                       tf.math.reduce_std(x_np_normalized)))

            x_tf_normalized = normalize_data(x_tf)
            self.assertAlmostEqual(tf.math.reduce_mean(x_tf_normalized).numpy(),
                                   0, 6,
                                   'mean = {}'.format(
                                       tf.math.reduce_mean(x_tf_normalized)))
            self.assertAlmostEqual(tf.math.reduce_std(x_tf_normalized).numpy(),
                                   1, 6,
                                   'stdev = {}'.format(
                                       tf.math.reduce_std(x_tf_normalized)))


    def test_runtime_comparison(self):
        """
        Run timeit to profile
        (1) np.utils pure numpy implementation,
        (2) tf implementation on numpy array
        (3) tf implementation on tf.Variable
        """

        def time_comparison(x_np, x_tf, num_repeats):

            print('Pure numpy implementation: ')
            start_time = timeit.default_timer()
            for ii in range(num_repeats): np_impl.utils.normalize_data(x_np)
            print(timeit.default_timer() - start_time)

            print('Tensorflow implementation on numpy array: ')
            start_time = timeit.default_timer()
            for ii in range(num_repeats): normalize_data(x_np)
            print(timeit.default_timer() - start_time)

            print('Tensorflow implementation on tf.Variable: ')
            start_time = timeit.default_timer()
            for ii in range(num_repeats): normalize_data(x_tf)
            print(timeit.default_timer() - start_time)

        print('Profile normalize_data(), matrix size = (16, 32, 64)')
        x_np = np.random.rand(16,32,64)
        x_tf = tf.Variable(x_np)
        num_repeats = 1000
        time_comparison(x_np, x_tf, num_repeats)

        # WARNING: this takes forever to run on the laptop
        # print('Profile normalize_data(), matrix size = (512, 512, 1024)')
        # x_np = np.random.rand(512,512,1024)
        # x_tf = tf.Variable(x_np)
        # num_repeats = 10
        # time_comparison(x_np, x_tf, num_repeats)



class TestImageToPoints(unittest.TestCase):
    """Test image_to_points function, using mnist images stored as a numpy array
    or a tf.Variable."""

    def _load_mnist_samples(self):
        """Load local MNIST samples x_train_*.npy."""
        num_samples = 30
        samples = [np.load('mnist_samples/x_train_%d.npy' % it) for it in
                   range(num_samples)]
        return samples


    def _test_image_to_points_np(self, images):
        """Test image_to_points with numpy.array input."""
        num_tests = 1
        thresholds = [60, 250]
        num_points = [128, 512, 1024]

        # First test default threshold and num_points
        for ii in range(num_tests):
            image = random.choice(images)
            image_to_points(image=image, viz=True)

        # Test various thresholds and num_points
        for ii in range(num_tests):
            for th in thresholds:
                for points in num_points:
                    image = random.choice(images)
                    image_to_points(image=image, threshold=th, num_points=points,
                                    viz=True)


    def _test_image_to_points_tf(self, images):
        """Test image_to_points with tf.Variable input."""
        pass


    def test_image_to_points(self):
        samples = self._load_mnist_samples()
        self._test_image_to_points_np(samples)



if __name__ == '__main__':
    unittest.main()
