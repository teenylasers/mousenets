import numpy as np


def normalize_data(x):
    """Normalize x, x can be a scalar, a vector, or a matrix."""
    # Normalize the mean
    norm_mean = np.mean(x)
    x = x - norm_mean
    # Normalize the variance
    norm_factor = np.sqrt(np.sum(np.square(x)) / x.size)
    assert norm_factor != 0
    return x * 1.0 / norm_factor


def image_to_points(array, threshold=128, num_points=256, viz=False):
    """Construct 2D point cloud representation for a grayscale image, keeping
    all pixels with a value greater than threshold. Delete or pad points so the
    result contains num_points. array is the input image as a nunmpy array. If
    viz, show the result as a scatter plot and the original data as an image."""

    dims = x_train[i].shape

    # Get points by the threshold
    mask = x_train[i] >= threshold
    s = np.empty((0,3), int)
    for j in range(dims[0]):
        for k in range(dims[1]):
            if mask[j][k] != 0:
                s = np.vstack((s, [j, k, array[j][k]]))

    # Delete or pad points to get num_points
    size_diff = s.shape[0] - num_points
    if size_diff > 0:
        for it in range(size_diff):
            reject_row = np.random.randint(0,s.shape[0])
            s = np.delete(s, (reject_row), axis=0)
    elif size_diff < 0:
        for it in range(-size_diff):
            add_row = np.random.randint(0,s.shape[0])
            s = np.vstack((s, s[add_row]))
    else:
        pass
    assert(s.shape[0] == num_points), 's.shape = {}, num_points = {}'.format(
        s.shape, num_points)

    # Visualize the 2D point cloud and the original image
    if viz:
        plt.imshow(x_train[i])
        plt.show()
        plt.scatter(s[:,1], -s[:,0], s=s[:,2]/10)
        plt.xlim(0, dims[1])
        plt.ylim(-dims[0], 0)
        plt.show()
        print(s.shape)

    # Shuffle up the points, so they are not ordered by x or y.
    np.random.shuffle(s)

    return s
