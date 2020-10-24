import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import constants


def convolution_im2col(x, kernel, s=1, p=0, bias=None):
    """
    Use im2col (tf.image.extract_patches()) to implement convolution.
    ns = num samples
    nc = num input channels
    c = num kernel channels, i.e. num output channels

    x = input, (ns * sample_dimensions), e.g. (ns * nx * ny * nc)
    s = stride
    p = zero padding width
    kernel = the convolution kernel, (kx * ky * nc * c)
    bias = add a constant to all output pixels after convolution
    """
    rates = [1 for it in x.shape]
    strides = [1 for it in x.shape]
    strides[1] = s
    strides[-2] = s
    tf.images.extract_patches(images=x, size=kernel.shape, strides=strides,
                              rates=rates, padding='VALID')



def convolution_naive(x, kernel, s=1, p=0, bias=None):
    """
    Naive nested for-loops implementation of convolution.
    ns = num samples
    nc = num input channels
    c = num kernel channels, i.e. num output channels

    x = input, (ns * sample_dimensions), e.g. (ns * nx * ny * nc)
    s = stride
    p = zero padding width
    kernel = the convolution kernel, (kx * ky * nc * c)
    bias = add a constant to all output pixels after convolution
    """
    ns,nx,ny,nc = x.shape
    kx,ky,ignore,c = kernel.shape
    assert (kernel.shape[1] == nc), 'kernel.shape (c * nc * kx * ky) = {}, '\
        'x.shape (ns * nc * nx * ny) = {}, nc mismatch'.format(
            kernel.shape, x.shape)

    if s is None:
        s = 1
    if p is None:
        p = 0
    if bias is None:
        bias = [0]*c

    # Initialize output
    nxo = int((nx + 2*p - kx)/s + 1)
    nyo = int((ny + 2*p - ky)/s + 1)
    # Have to use np, because tf EagerTensor does not support item assignment.
    y = np.zeros([ns, nxo, nyo, c])

    # Naive convolution
    xp = zero_padding(x, p)
    for si, ci, xi, yi in itertools.product(*map(range, (ns, c, nxo, nyo))):
        sub_x = xp[si,
                   xi*s : xi*s+kx,
                   yi*s : yi*s+ky,
                   :]
        y[si, xi, yi, ci] = tf.reduce_sum(sub_x * kernel[:,:,:,ci]) + bias[ci]

    # Return results in float32, consistent with tf default implementation.
    return tf.cast(tf.Variable(y), tf.float32)


def zero_padding(x, p, skip_dims=True):
    """
    x = input, (num_samples * sample_dimensions), e.g. (ns * nc * nx * ny)
    p = zero padding width
    skip_dims = skip the first dimensions, e.g. if they represent multiple
    samples and/or multiple channels. Default assumes multiple samples and
    multiple channels.
    """
    skip = [[0,0]]
    padding = [[p,p] for it in range(tf.rank(x)-2)] # skip first and last dims
    if skip_dims:
        padding = tf.concat([skip,padding,skip], axis=0)
    xp = tf.pad(x, padding, 'CONSTANT')
    return xp
