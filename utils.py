import random, itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import constants


def normalize_data(x):
    """Normalize x, x can be a vector or a matrix."""
    data_size = tf.cast(tf.size(x), tf.float64)
    assert(data_size > 1)
    # Normalize the mean
    norm_mean = tf.math.reduce_mean(x)
    x = x - norm_mean
    # Normalize the variance
    norm_factor = tf.math.reduce_std(x)
    assert norm_factor != 0
    return x * 1.0 / norm_factor


def image_to_points(image, threshold=128, num_points=256, viz=False):
    """Construct 2D point cloud representation for a grayscale image, keeping
    all pixels with a value greater than threshold. Delete or pad points so the
    result contains num_points. image is the input image as a numpy array. If
    viz, show the result as a scatter plot and the original data as an image."""

    dims = image.shape
    if num_points > image.size:
        print('Warning: num_points [{}] > num pixels in image [{}], use {}.'.format(
            num_points, image.size, image.size))
        num_points = image.size

    # Get points by the threshold
    mask = tf.constant(image >= threshold)
    # Start a first row of zeros to concatenate points to
    s = [[j, k, image[j][k]] for j in range(dims[0]) for k in range(dims[1])
         if mask[j][k]]

    # Delete or pad points to get num_points
    size_diff = len(s) - num_points
    if size_diff > 0:
        s = s[:-size_diff]
    elif size_diff < 0:
        for it in range(-size_diff):
            s.append(random.choice(s))
    else:
        pass

    # Shuffle up the points, so they are not ordered by x or y.
    s = tf.stack(s)
    s = tf.random.shuffle(s)
    assert(s.shape[0] == num_points), 's.shape = {}, num_points = {}'.format(
        s.shape, num_points)

    # Visualize the 2D point cloud and the original image
    if viz:
        plt.imshow(image)
        plt.show()
        plt.scatter(s[:,1], -s[:,0], s=s[:,2]/10)
        plt.xlim(0, dims[1])
        plt.ylim(-dims[0], 0)
        plt.show()

    return s


def compare_matrices(a, b, err=constants.kAllowNumericalErr):
    """Compare 2 matrices, a and b, elementwise. Allow an error to account for
    numerical accuracy. Return true if the elementwise difference between a
    and b are all less than err."""

    assert a.shape == b.shape, 'a.shape {} != b.shape {}'.format(
        a.shape, b.shape)
    size = tf.reduce_prod(a.shape)

    # Reshape into 1D vector before comparison, to be agnostic of input ndims
    a = tf.reshape(a, [size])
    b = tf.reshape(b, [size])
    for ii in range(size):
        t = a[ii] - b[ii]
        if abs(t) > err:
            return False
    return True
