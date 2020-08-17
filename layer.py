import numpy as np
import abc

from constants import *


class Layer(object):
  """A layer in a neural network."""
  __metaclass__ = abc.ABCMeta


  @abc.abstractmethod
  def forward_pass(self, x, save, w=None, b=None):
    return


  @abc.abstractmethod
  def backprop(self, dLdy, save):
    return


  def check_gradient(self, dLdy):
    """Assume we have performed backprop to update self.dLdx and self.dLdw,
    check them against numerical calculations from the same dLdy."""
    # dLdx, dLdw = analytical gradient
    dLdx, dLdw = backprop(dLdy, save=False)

    return self._check_gradient_dLdx(dLdy, dLdx) and \
      self._check_gradient_dLdw(dLdy, dLdw)


  @abc.abstractmethod
  def _check_gradient_dLdx(self, dLdy, dLdx):
    return


  @abc.abstractmethod
  def _check_gradient_dLdw(self, dLdy, dLdw):
    return




class DenseLayer(Layer):
  """
  A fully connected layer, y = f(h) = f(w.dot(x)), where f is an activation
  function, either sigmoid or softmax.
  """
  # Definitions:

  # n = layer width
  # x = input vector
  # nx = input vector length
  # w = weights matrix
  # h = intermediate variable, w.dot(x)
  # y = output vector

  # activation = activation function name
  # f = the activation function
  # dfdh = the derivative of the activation function

  # dLdx = backpropagation from dLdy to dLdx
  # dLdw = backpropagation from dLdy to dLdw

  def __init__(self, nx, n, activation, w=None):

    assert(isinstance(n, int) and n > 0), 'n should be an integer and >0.'
    assert(isinstance(nx, int) and nx > 0), 'nx should be an integer and >0.'

    self.nx = nx
    self.n = n
    self.activation = activation
    if self.activation == 'sigmoid':
      self.f = self._f_sigmoid
      self.dfdh = self._dfdh_sigmoid
    elif self.activation == "softmax":
      self.f = self._f_softmax
      self.dfdh = self._dfdh_softmax
    else:
      print("Error: activation function %s is not implemented.",
            self.activation)

    # Initialize w if not provided
    if w is None:
      self.w = self._initialize_weights(self.n, self.nx)
    else:
      assert(w.shape == (self.n, self.nx)), \
        'User input w has the wrong dimensions {}'.format(w.shape)
      self.w = w

    # Initialize/reset the remaining states of this layer: self.x, h, y, dLdx, dLdw
    self.reset_cache()


  def _initialize_weights(self, n, m):
    """Initialize a weights matrix of dimensions (n x m)."""
    return np.random.rand(n, m) * 2 - 1


  def reset_cache(self):
    """Clear cached states. self.x, h, y, dLdw, and dLdy"""
    self.x = None
    self.h = None
    self.y = None
    # For batch training, we accumulate dLdw from each sample and update w at
    # the end of the batch.
    self.dLdw = np.zeros((self.n, self.nx))
    self.dLdy = None


  def forward_pass(self, x, save, w=None, b=None):
    """Perform forward pass on this layer using input x, return the output y.
    If save, then update self.x, self.y, self.h."""

    assert(isinstance(x, np.ndarray)), 'x should be a numpy array.'
    assert(len(x.shape)==1), 'x should be a vector.'
    assert(x.shape[0]==self.nx), 'x should have length %d, not %d.' % \
      (self.nx, x.shape[0])
    if w is not None:
      assert(not save), \
        'When user specifies a weights matrix w, cannot save the results to '\
        'modify the layer.'
    assert(b is None), "Bias is incorporated in the weight matrix, b is unused."

    if w is None:
      h = self.w.dot(x)
    else:
      h = w.dot(x)

    y = self.f(h)

    if save:
      self.x = x
      self.h = h
      self.y = y

    # Return output of this layer
    return y


  def backprop(self, dLdy, save):
    """Calculate and return (dLdx, dLdw). If save, append this dLdw to
    self.dLdw, and save self.dLdy."""
    # dLdy = (n * 1)
    # dydh = dfdh = (n * n) diagonal
    # dhdx = (n * nx) = w
    # dhdw = (nx * 1) = x
    # dLdw = dydh * dLdy * dhdw.T
    # dLdx = dhdx.T * dydh * dLdy

    # Do backprop
    dydh = self.dfdh(self.h)
    if len(self.x.shape) == 1:
      # numpy does not distinguish between row and column vectors, use np.outer
      dLdw = np.outer(dydh @ dLdy, self.x.T)
    else:
      dLdw = dydh @ dLdy @ self.x.T
    dLdx = self.w.T @ dydh @ dLdy
    if save:
      self.dLdx = dLdx
      self.dLdw = self.dLdw + dLdw
      self.dLdy = dLdy # Cache dLdy for gradient checking

    # Return (dLdx, dLdw) for the previous layer's backprop
    return dLdx, dLdw


  def update_weights(self, dLdw, batch_size):
    """Given a dLdw and the batch_size that accumulated it, update self.w."""
    self.w = self.w - self.dLdw / batch_size


  def check_gradient(self, dLdy):
    """Assume we have performed backprop to update self.dLdx and self.dLdw,
    check them against numerical calculations from the same dLdy."""
    # dLdx, dLdw = analytical gradient
    dLdx, dLdw = self.backprop(dLdy, save=False)

    return self._check_gradient_dLdx(dLdy, dLdx) and \
      self._check_gradient_dLdw(dLdy, dLdw)


  def _check_gradient_dLdx(self, dLdy, dLdx):
    """Check the backprop analytical gradient dLdx against the equivalent
    numerical calculation."""
    # dydx = [y_j(x + eps) - y_j(x - eps)] / (2*eps)
    # dLdx = \sum_j dLdy_j * dydx

    x_dims = self.x.shape

    # g = numerical gradient
    g = np.zeros(x_dims)
    for i in range(x_dims[0]):
      x_eps = np.zeros(x_dims)
      x_eps[i] = kEpsilonNumericalGrad
      dydx = (self.forward_pass(self.x + x_eps, save=False) - \
              self.forward_pass(self.x - x_eps, save=False))  \
              / 2.0 / kEpsilonNumericalGrad
      g[i] = dLdy.dot(dydx)

    # Print output.
    # print("Analytical dLdx = \n%r" % dLdx)
    # print("Numerical dLdx = \n%r" % g)
    # print("Max (analytical - numerical) error = %r" % np.max(dLdx - g))
    # print("Min (analytical - numerical) error = %r" % np.min(dLdx - g))

    # Return check results
    return np.max(np.square(g-dLdx)) < kAllowNumericalErr


  def _check_gradient_dLdw(self, dLdy, dLdw):
    """Check the backprop analytical gradient dLdw against the equivalent
    numerical calculation."""
    # dydw_j = [y_j(w + eps) - y_j(w - eps)] / (2*eps)
    # dLdw = \sum_j dLdy_j * dydw_j

    y_dims = self.y.shape
    w_dims = self.w.shape

    # g = numerical loss gradient
    g = np.zeros(w_dims)
    dydw = np.zeros(y_dims + w_dims)
    for wi in range(w_dims[0]):
      for wj in range(w_dims[1]):
        w_eps = np.zeros(w_dims)
        w_eps[wi, wj] = kEpsilonNumericalGrad
        dydw = (self.forward_pass(self.x, save=False, w = self.w + w_eps) - \
                self.forward_pass(self.x, save=False, w = self.w - w_eps))  \
                / 2.0 / kEpsilonNumericalGrad
        g[wi, wj] = dLdy.dot(dydw)

    # Print output
    # print("Analytical dLdw = \n%r" % dLdw)
    # print("Numerical dLdw = \n%r" % g)
    # print("Max (analytical - numerical) error = %r" % np.max(dLdw - g))
    # print("Min (analytical - numerical) error = %r" % np.min(dLdw - g))

    # Return check results
    return np.max(np.square(g-dLdw)) < kAllowNumericalErr


  def get_input_size(self):
    return self.nx


  def get_width(self):
    return self.n


  def get_weights(self):
    return self.w


  def _f_sigmoid(self, h):
    """Evaluate the sigmoid function for h, where h is a vector or a matrix."""
    return 1 / (1 + np.exp(-h))


  def _dfdh_sigmoid(self, h):
    """Evaluate the gradient of sigmoid function at h, where h is a vector."""
    f_h = self._f_sigmoid(h)
    return np.diag((1 - f_h)*f_h)


  def _f_softmax(self, h):
    """Evaluate the softmax function for h, where h is a vector."""
    assert(len(h.shape)==1), 'Input arg h should be a vector.'
    exp_h = np.exp(h)
    return exp_h/np.sum(exp_h)


  def _dfdh_softmax(self, h):
    """Evaluate the gradient of softmax function at h, where h is a vector."""
    assert(len(h.shape)==1), 'Input arg h should be a vector.'
    f_h = self._f_softmax(h)
    dfdh = np.diag(f_h) - np.outer(f_h, f_h)
    return dfdh


class ConvLayer(Layer):
  """
  A convolutional layer.
  """
  # Definitions:

  # x = input image, a 3D matrix (nc * nx * ny)
  # y = output image, a 3D matrix (c * nxo * nyo)

  # nx, ny = input image dimensions
  # nc = input image num channels
  # nxo, nyo = (nx-k+2p)/s+1, (ny-k+2p)/s+1

  # k = kernel size
  # c = num filters in this layer
  # s = stride
  # p = num pixels for zero-padding

  # w = kernels = (c * nc * k * k) matrix
  # xp = padded input image, (nc * (nx+2p) * (ny+2p)) matrix

  def __init__(self, nx, ny, nc, k, c, s, p, w=None):

    assert(isinstance(nx, int) and nx > 0 and isinstance(ny, int) and ny > 0 \
           and isinstance(nc, int) and nc > 0)
    assert(isinstance(k, int) and k > 0 and isinstance(c, int) and c > 0)
    assert(isinstance(s, int) and s > 0 and isinstance(p, int) and p > 0)

    self.nx = nx
    self.ny = ny
    self.nc = nc
    self.k = k
    self.c = c
    self.s = s
    self.p = p

    self.nxo = (nx + 2*p - k)/s + 1
    self.nyo = (ny + 2*p - k)/s + 1

    # Initialize kernels
    if w is None:
      self.w = self._initialize_kernel(self.k, self.nc, self.c)
    else:
      assert(w.shape == (self.c, self.nc, self.k, self.k)), \
        'User input w has the wrong dimensions {}'.format(w.shape)
      self.w = w

    # Initialize/reset the remaining states of this layer
    self.reset_cache()


  def _initialize_kernel(self, k, nc, c):
    """Initialize a kernel matrix of dimensions (k * k * nc)."""
    return np.random.rand(c, nc, k, k) * 2 - 1


  def reset_cache(self):
    """Clear cached states. self.x, y, dLdw, dLdb, and dLdy"""
    self.x = None
    self.y = None
    self.dLdw = np.zeros((self.c, self.nc, self.k, self.k))
    self.dLdb = np.zeros((self.nc, self.nx, self.ny))
    self.dLdy = None


  def forward_pass(self, x, save, w=None, b=None):
    """Perform forward pass on this layer using input x, return output y. If
    save, then update self.x and self.y."""

    # Sanity check input
    assert(isinstance(x, np.ndarray)), 'x should be a numpy array.'
    assert(len(x.shape)==3), 'x should be a 3D matrix.'
    assert(x.shape[0]==self.nc and x.shape[1]==self.nx and x.shape[2]==self.ny), \
      'x should have dimensions (%d,%d,%d), not %r' % \
      (self.nc, self.nx, self.ny, x.shape)
    if w is not None:
      assert(not save), \
        'When user specifies w, cannot save the results to modify the layer '\
        'property.'

    # Form w and b, the kernel and associated bias to use in this forward prop.
    if w is None:
      w = self.w
    else:
      assert(w.shape == (nc, k, k))
    if b is None:
      b = self.b
    else:
      assert(b.shape == (nc, 1))

    # Zero-padding
    xp = np.pad(x, ((0, 0),(self.p, self.p),(self.p, self.p)), 'constant')
    # TODO: delete the following asserts after debug
    assert(xp.shape[0]==self.nc and xp.shape[1]==self.nx+2*self.p \
           and xp.shape[2]==self.ny+2*self.p), \
           'zero-padded x has the wrong dimensions: %r' % xp.shape

    # Initialize output matrix
    y = np.zeros((self.c, self.nxo, self.nyo))

    # Perform convolution
    for ci in range(self.c):
      for xi in range(self.nxo):
        for yi in range(self.nyo):
          sub_x = xp[:,
                     xi*self.s:xi*self.s+self.k,
                     yi*self.s:yi*self.s+self.k]
          y[ci, xi, yi] = np.sum(sub_x)

    # Save results if needed
    if save:
      self.xp = xp
      self.y = y

    # Return results
    return y


  def backprop(self, dLdy, save):
    """Calculate and return (dLdx, dLdw, dLdb). If save, append this dLdw to
    self.dLdw, this dLdb to self.dLdb, and save self.dLdy."""

    # dLdy = (c * nxo * nyo)
    # dLdw = (c * nc * k * k)
    dLdw = np.zeros((self.c, self.nc, self.k, self.k))
    for ci in range(self.c): # output channel index
      for nci in range(self.nc): # input channel index
        for kxi in range(self.k): # kernel x index
          for kyi in range(self.k): # kernel y index
            for xi in range(self.nxo): # output x index
              for yi in range(self.nyo): # output y index
                dLdw[ci, nci, kxi, kyi] += dLdy[ci, xi, yi] * \
                  self.xp[nci,
                          self.s * xi + self.kxi - 1,
                          self.s * yi + self.kyi - 1]

    # dLdb = (ci * 1)
    dLdb = np.zeros((self.ci, 1))
    for ci in range(self.c):
      for xi in range(self.nxo):
        for yi in range(self.nyo):
          dLdb[ci] += dLdy[ci, xi, yi]

    # dLdx = (nc * nx * ny)
    dLdx = np.zeros((self.nc, self.nx, self.ny))
    for nci in range(self.nc): # input channel index
      for xi in range(self.nx): # input x index
        for yi in range(self.ny): # input y index
          for xoi in range(self.nxo): # output x index
            for yoi in range(self.nyo): # output y index
              for ci in range(self.c):
                dLdx[nci, xi, yi] += dLdy[ci, xoi, yoi] * \
                  self.w[ci,
                         xi + 1 - self.s * xoi,
                         yi + 1 - self.s * yoi]

    # Save results if needed
    if save:
      self.dLdx = dLdx
      self.dLdw = self.dLdw + dLdw
      self.dLdb = self.dLdb + dLdb
      self.dLdy = dLdy # Cache dLdy for gradient checking

    # Return (dLdx, dLdw, dLdb) for the previous layer's backprop
    return (dLdx, dLdw, dLdb)


class ActivationLayer(Layer):
  """
  An activation layer that takes a 3D matrix as input.
  """
  # Definitions:

  # x = input image, a 3D matrix (nc * nx * ny)
  # y = output image, a 3D matrix (nc * nx * ny)

  # nx, ny = image dimension, same for input and output
  # nc = image num channels, same for input and output

  # activation = activation function name
  # f = the activation function
  # dfdx = the derivative of the activation function

  # dLdx = backpropagation from ddy to dLdx

  def __init__(self, nx, ny, nc, activation):

    assert(isinstance(nx, int) and nx > 0), 'nx should be an integer and >0.'
    assert(isinstance(ny, int) and ny > 0), 'ny should be an integer and >0.'
    assert(isinstance(nc, int) and nc > 0), 'nc should be an integer and >0.'

    self.activation = activation
    if self.activation == 'relu':
      self.f = self._f_relu
      self.dfdx = self._dfdx_relu
    else:
      assert False, "Error: activation function %s is not implemented." % \
        self.activation


  def forward_pass(self, x, save, w=None, b=None):
    """Perform forward pass on this layer using input x, return the output y.
    If save, then update self.x and self.y."""

    assert(w is None and b is None), \
      'w and b should be None, they are N/A for this layer.'
    assert(isinstance(x, np.ndarray)), 'x should be a numpy array.'
    assert(x.shape == (nc, nx, ny)), 'x should have the shape %r.' % x.shape
    y = self.f(x)

    if save:
      self.x = x
      self.y = y

    return y


  def backprop(self, dLdy, save):
    """Calculate and return dLdx. If save, update self.dLdy."""
    dLdx = dLdy * self.dfdx(self.x)
    if save:
      self.dLdy = dLdy
    return dLdx


  def _f_relu(self, x):
    """Implement ReLU function, return ReLU(x)"""
    return x * (x > 0)


  def _dfdx_relu(self, x):
    """Implement the gradient of ReLU function, return del ReLU(x)"""
    return (x > 0) * 1


class PoolingLayer(Layer):
  """
  A pooling layer.
  """
  # Definitions:

  # x = input image, a 3D matrix (nc * nx * ny)
  # y = output image, a 3D matrix (nc * nxo * nyo)

  # nx, ny = input image dimensions
  # nc = image num channels, same for input and output images
  # nxo, nyo = (nx-k)/s+1, (ny-k)/s+1

  # k = kernel size
  # s = stride
  # operator = pooling operator, take the max or average over the kernel


  def __init__(self, nx, ny, nc, k, s, operator):

    assert(isinstance(nx, int) and nx > 0 and isinstance(ny, int) and ny > 0 \
           and isinstance(nc, int) and nc > 0)
    assert(isinstance(k, int) and k > 0 and isinstance(c, int) and s > 0)

    self.nx = nx
    self.ny = ny
    self.nc = nc
    self.k = k
    self.s = s

    self.nxo = (nx - k)/s + 1
    self.nyo = (ny - k)/s + 1

    self.operator = operator
    if self.operator == 'max':
      self.f = self._f_max
      self.dfdx = self._dfdx_max
    elif self.operator == 'average':
      self.f = self._f_average
      self.dfdx = self._dfdx_average
    else:
      assert False, 'Error: pooling operator %s is not implemented.' % operator


  def forward_pass(self, x, save, w=None, b=None):
    """Perform forward pass, return output y. If save, then update self.x and
    self.y"""

    # Sanity check input
    assert(w is None and b is None), \
      'w and b should be None, they are N/A for this layer.'
    assert(isinstance(x, np.ndarray) and x.shape==(self.nc, self.nx, self.ny))

    # Perform pooling
    y = np.zeros((self.nc, self.nxo, self.nyo))
    for ci in range(self.nc):
      for xi in range(self.nxo):
        for yi in range(self.nyo):
          sub_x = x[ci, xi*s:xi*s+k, yi*s:yi*s+k]
          y[ci, xi, yi] = self.f(x)

    # If save
    if save:
      self.x = x
      self.y = y

    # Return output
    return y


  def backprop(self, dLdy, save):
    """Calculate and return dLdx. If save, update self.dLdy."""

    # dLdy = (nc, nxo, nyo)
    # dLdx = (nc, nx, ny)
    dLdx = np.zeros(self.nc, self.nx, self.ny)
    for ci in range(self.nc):
      for xii in range(self.nx):
        for yii in range(self.ny):
          for xoi in range(self.nxo):
            for yoi in range(self.nyo):
              sx = self.x[ci,
                          xoi * self.s + self.k,
                          yoi * self.s + self.k]
              dLdx[ci, xii, yii] += dLdy[ci, xoi, yoi] * self.dfdx(sx)

    # If save
    if save:
      self.dLdx = dLdx

    # Return
    return dLdx


  def _f_max(self, sx):
    """Return the maximum element in the input matrix sx, a pooling submatrix of
    x."""
    return np.max(sx)


  def _dfdx_max(self, sx):
    """Return the gradient of the max pooling function for an input matrix sx,
    where sx is a submatrix of x and is the size of the pooling kernel."""
    max_sx = np.max(sx)
    return (sx == max_sx) * 1


  def _f_average(self, sx):
    """Return the mean value of all elements in an input matrix sx, a pooling
    submatrix of x."""
    return np.mean(sx)


  def _dfdx_average(self, sx):
    """Return the gradient of the average pooling functions for an input matrix
    sx, where sx is a submatrix of x and is the size of the pooling kernel."""
    return sx * (1/self.k/self.k)