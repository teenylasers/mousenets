import numpy as np
import abc, itertools

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


  def update_weights(self, batch_size, dLdw, dLdb):
    """Given a dLdw and the batch_size that accumulated it, update self.wb."""
    self.w = self.w - self.dLdw / batch_size
    self.b = self.b - self.dLdb / batch_size


  def check_gradient(self, dLdy):
    """Assume we have performed backprop to update self.dLdx and self.dLdw,
    check them against numerical calculations from the same dLdy."""
    # dLdx, dLdw = analytical gradient
    dLdx, dLdw, dLdb = self.backprop(dLdy, save=False)

    return self._check_gradient_dLdx(dLdy, dLdx) and \
      self._check_gradient_dLdw(dLdy, dLdw, dLdb)


  @abc.abstractmethod
  def _check_gradient_dLdx(self, dLdy, dLdx):
    """Check the backprop analytical gradient dLdx against the equivalent
    numerical calculation."""
    return


  @abc.abstractmethod
  def _check_gradient_dLdw(self, dLdy, dLdw=None, dLdb=None):
    """Check the backprop analytical gradient dLdw (and dLdb for some layer
    implementations) against the equivalent numerical calculation."""
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
  # w = weights matrix, (n+1) * nx
  # h = intermediate variable, w.dot(x)
  # y = output vector

  # activation = activation function name
  # f = the activation function
  # dfdh = the derivative of the activation function

  # dLdx = backpropagation from dLdy to dLdx
  # dLdw = backpropagation from dLdy to dLdw

  def __init__(self, nx, n, activation, w=None, b=None):

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
    elif self.activation == "relu":
      self.f = self._f_relu
      self.dfdh = self._dfdh_relu
    else:
      print("Error: activation function %s is not implemented.",
            self.activation)

    # Initialize w and b if not provided
    if w is None:
      self.w = self._initialize_weights(self.n, self.nx)
    else:
      assert(w.shape == (self.n, self.nx)), \
        'User input w has the wrong dimensions {}'.format(w.shape)
      self.w = w
    if b is None:
      self.b = self._initialize_bias(self.n)
    else:
      assert(b.shape == (self.n, 1)), \
        'User input b has the wrong dimensions {}'.format(b.shape)
      self.b = b

    # Concatenate w and b to self.wb for forward_pass and backprop calculations
    self.wb = self._concat_w_b(self.w, self.b)

    # Initialize/reset the remaining states of this layer: self.x, h, y, dLdx, dLdw
    self.reset_cache()


  def _initialize_weights(self, n, m):
    """Initialize a weights matrix of dimensions (n x m)."""
    return np.random.rand(n, m) * 2 - 1


  def _initialize_bias(self, n):
    """Initialize a bias vector of dimensions (1 x n)."""
    return np.random.rand(n, 1) * 2 - 1


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
    if w is not None or b is not None:
      assert(not save), \
        'When user specifies a weights matrix w and/or a bias vector b, cannot' \
        ' save the results to modify the layer.'

    # Apply user supplied w and b, if provided
    if w is None:
      w = self.w
    if b is None:
      b = self.b
    wb = self._concat_w_b(w,b)

    # Decide whether the input x already has the bias term attached.
    if len(x) == self.nx:
      x = np.append(x,1)
    else:
      assert len(x) == self.nx + 1

    h = wb.dot(x)
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

    dydh = self.dfdh(self.h)

    # Backprop for dLdw (whose last row is dLdb)
    if len(self.x.shape) == 1:
      # numpy does not distinguish between row and column vectors, use np.outer
      dLdwb = np.outer(dydh @ dLdy, self.x.T)
    else:
      assert False, 'Where to append 1 to x for bias, and does this work for 2D x?'
      dLdwb = dydh @ dLdy @ self.x.T

    # Split dLdwb into dLdw and dLdb to be consistent with the other layers
    dLdw = dLdwb[:, :-1]
    dLdb = dLdwb[:, -1:]

    # Backprop for dLdx
    dLdx = self.wb.T @ dydh @ dLdy
    dLdx = dLdx[:-1] # The last term is for the bias, no need to backpropagate

    # If cache the gradients
    if save:
      self.dLdx = dLdx
      self.dLdw = self.dLdw + dLdw
      self.dLdb = self.dLdb + dLdb
      self.dLdy = dLdy # Cache dLdy for gradient checking

    # Return (dLdx, dLdw, dLdb) for the previous layer's backprop
    return dLdx, dLdw, dLdb


  def update_weights(self, batch_size, dLdw, dLdb):
    super(DenseLayer, self).update_weights(batch_size, dLdw, dLdb)
    self.wb = self._concat_w_b(self.w, self.b)


  def _concat_w_b(self, w, b):
    """Helper function to concatenate w matrix and b vector, so we can use a single
    matrix operation for both w and b in forward_pass and backprop."""
    return np.c_[w, b]


  def _check_gradient_dLdx(self, dLdy, dLdx):

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


  def _check_gradient_dLdw(self, dLdy, dLdw, dLdb):

    # dydw_j = [y_j(w + eps) - y_j(w - eps)] / (2*eps)
    # dLdw = \sum_j dLdy_j * dydw_j

    y_dims = self.y.shape
    wb_dims = self.wb.shape

    # g = numerical loss gradient for weights and biases
    g = np.zeros(wb_dims)
    for wi in range(wb_dims[0]):
      for wj in range(wb_dims[1]):
        wb_eps = np.zeros(wb_dims)
        wb_eps[wi, wj] = kEpsilonNumericalGrad
        wb_nudge_up = self.wb + w_eps
        wb_nudge_down = self.wb - w_eps
        dydwb = (self.forward_pass(self.x, save=False,
                                   w = wb_nudge_up[:,:-1],
                                   b = wb_nudge_up[:,-1:]) - \
                 self.forward_pass(self.x, save=False,
                                   w = wb_nudge_down[:,:-1],
                                   b = wb_nudge_down[:,-1:])) \
                 / 2.0 / kEpsilonNumericalGrad
        g[wi, wj] = dLdy.dot(dydwb)

    # Print output
    # print("Analytical dLdw = \n%r" % dLdw)
    # print("Numerical dLdw = \n%r" % g)
    # print("Max (analytical - numerical) error = %r" % np.max(dLdw - g))
    # print("Min (analytical - numerical) error = %r" % np.min(dLdw - g))

    # Return check results
    gw = g[:,:-1]
    gb = g[:,-1]
    return np.max(np.square(gw - dLdw)) < kAllowNumericalErr and \
      np.max(np.square(gb - dLdb)) < kAllowNumericalErr


  def get_input_size(self):
    return self.nx


  def get_width(self):
    return self.n


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


  def _f_relu(self, h):
    """Evaluate the ReLU function for h."""
    return h * (h > 0)


  def _dfdh_relu(self, h):
    """Evaluate the gradient of ReLU function at h."""
    return np.diag((h > 0) * 1)



class DenseLayer2D(DenseLayer):
  """
  A DenseLayer that accepts a 2D or 3D image as input. It reshapes the 2D image
  into a vector and thereafter functions as a DenseLayer.
  """

  def __init__(self, nx, ny, nc, n, activation, w=None, b=None):

    assert(isinstance(nx, int) and nx > 0 and isinstance(ny, int) and ny > 0
           and isinstance(nc, int) and nc > 0)
    super(DenseLayer2D, self).__init__(nx * ny * nc, n, activation, w, b)


  def forward_pass(self, x, save, w=None, b=None):
    x = x.reshape(self.nx)
    return super(DenseLayer2D, self).forward_pass(x, save, w, b)



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

  def __init__(self, nx, ny, nc, k, c, s, p, w=None, b=None):

    assert(isinstance(nx, int) and nx > 0 and isinstance(ny, int) and ny > 0 \
           and isinstance(nc, int) and nc > 0), \
           'nx = %d, ny = %d, nc = %d' % (nx, ny, nc)
    assert(isinstance(k, int) and k > 0 and isinstance(c, int) and c > 0)
    assert(isinstance(s, int) and s > 0 and isinstance(p, int) and p >= 0)

    self.nx = nx
    self.ny = ny
    self.nc = nc
    self.k = k
    self.c = c
    self.s = s
    self.p = p

    self.nxo = int((nx + 2*p - k)/s + 1)
    self.nyo = int((ny + 2*p - k)/s + 1)

    assert(self.nxo > 0 and self.nyo > 0), \
      'self.nxo = %d, self.nyo = %d' % (self.nxo, self.nyo)

    # Initialize kernel and bias
    if w is None:
      self.w = self._initialize_kernel(self.k, self.nc, self.c)
    else:
      assert(w.shape == (self.c, self.nc, self.k, self.k)), \
        'User input w has the wrong dimensions {}'.format(w.shape)
      self.w = w
    if b is None:
      self.b = self._initialize_bias(self.c)
    else:
      assert(b.shape == (self.c, 1)), \
        'User input b has the wrong dimensions {}'.format(w.shape)

    # Initialize/reset the remaining states of this layer
    self.reset_cache()


  def _initialize_kernel(self, k, nc, c):
    """Initialize a kernel matrix of dimensions (k * k * nc)."""
    return np.random.rand(c, nc, k, k) * 2 - 1


  def _initialize_bias(self, c):
    """Initialize a bias vector of dimensions (c * 1)."""
    return np.random.rand(c, 1) * 2 - 1


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

    # Form w and b, the kernel and associated bias to use in this forward prop.
    if w is None:
      w = self.w
    else:
      assert(w.shape == (self.c, self.nc, self.k, self.k))
    if b is None:
      b = self.b
    else:
      assert(b.shape == (self.c, 1))

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
          y[ci, xi, yi] = np.sum(sub_x * w[ci]) + b[ci]

    # Save results if needed
    if save:
      self.x = x
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
                          self.s * xi + kxi,
                          self.s * yi + kyi]

    # dLdb = (ci * 1)
    dLdb = np.zeros((self.c, 1))
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
                if (xi + self.p - self.s * xoi < self.k) and \
                   (xi + self.p - self.s * xoi >= 0) and \
                   (yi + self.p - self.s * yoi < self.k) and \
                   (yi + self.p - self.s * yoi >= 0):
                  dLdx[nci, xi, yi] += dLdy[ci, xoi, yoi] * \
                    self.w[ci,
                           nci,
                           xi + self.p - self.s * xoi,
                           yi + self.p - self.s * yoi]

    # Save results if needed
    if save:
      self.dLdx = dLdx
      self.dLdw = self.dLdw + dLdw
      self.dLdb = self.dLdb + dLdb
      self.dLdy = dLdy # Cache dLdy for gradient checking

    # Return (dLdx, dLdw, dLdb) for the previous layer's backprop
    return (dLdx, dLdw, dLdb)


  def _check_gradient_dLdx(self, dLdy, dLdx):

    # self.x = (nc * nx * ny)
    x_dims = self.x.shape

    # g = numerical gradient
    g = np.zeros(x_dims)
    for cii, xii, yii in itertools.product(*[range(it) for it in x_dims]):
      x_eps = np.zeros(x_dims)
      x_eps[cii, xii, yii] = kEpsilonNumericalGrad
      dydx = (self.forward_pass(self.x + x_eps, save=False) - \
              self.forward_pass(self.x - x_eps, save=False))  \
              / 2.0 / kEpsilonNumericalGrad
      g[cii, xii, yii] = np.sum(dLdy * dydx)

    # Return check results
    res = np.max(np.square(g-dLdx)) < kAllowNumericalErr and \
      np.sum(g) != 0 and np.sum(dLdx) != 0
    if not res:
      print('Analytical dLdx = \n%r' % dLdx)
      print('Numerical dLdx = \n%r' % g)
      print('Max (analytical - numerical) error = %r' % np.max(dLdx - g))
      print('Min (analytical - numerical) error = %r' % np.min(dLdx - g))
      print('Error: %s check dLdx gradient failed.' % self.__class__)
    return res


  def _check_gradient_dLdw(self, dLdy, dLdw, dLdb):

    w_dims = self.w.shape # self.w = (c * k * k)
    b_dims = self.b.shape # self.b = (c

    # gw = numerical gradient equivalent for dLdw
    gw = np.zeros(w_dims)
    for ci in range(self.c):
      for nci in range(self.nc):
        for wi in range(self.k):
          for wj in range(self.k):
            w_eps = np.zeros(w_dims)
            w_eps[ci, nci, wi, wj] = kEpsilonNumericalGrad
            dydw = (self.forward_pass(self.x, save=False, w = self.w + w_eps) - \
                    self.forward_pass(self.x, save=False, w = self.w - w_eps))  \
                    / 2.0 / kEpsilonNumericalGrad
            gw[ci, nci, wi, wj] = np.sum(dLdy * dydw)

    # gb = numerical gradient equivalent for dLdb
    gb = np.zeros(b_dims)
    for ci in range(self.c):
      b_eps = np.zeros(b_dims)
      b_eps[ci] = kEpsilonNumericalGrad
      dydb = (self.forward_pass(self.x, save=False, w=None, b = self.b + b_eps) - \
              self.forward_pass(self.x, save=False, w=None, b = self.b - b_eps))  \
              / 2.0 / kEpsilonNumericalGrad
      gb[ci] = np.sum(dLdy * dydb)

    # Return check results
    dLdw_res = (np.max(np.square(gw-dLdw)) < kAllowNumericalErr)
    if not dLdw_res:
      print('Analytical dLdw = \n%r' % dLdw)
      print('Numerical dLdw = \n%r' % gw)
      print('Max (analytical - numerical) error = %r' % np.max(dLdw - gw))
      print('Min (analytical - numerical) error = %r' % np.min(dLdw - gw))
      print('Error: %s check dLdw gradient failed.' % self.__class__)

    dLdb_res = (np.max(np.square(gb-dLdb)) < kAllowNumericalErr)
    if not dLdb_res:
      print('Analytical dLdb = \n%r' % dLdb)
      print('Numerical dLdb = \n%r' % gb)
      print('Max (analytical - numerical) error = %r' % np.max(dLdb - gb))
      print('Min (analytical - numerical) error = %r' % np.min(dLdb - gb))
      print('Error: %s check dLdb gradient failed.' % self.__class__)

    return dLdw_res and dLdb_res


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

  # dLdx = backpropagation from dLdy to dLdx

  def __init__(self, nx, ny, nc, activation):

    assert(isinstance(nx, int) and nx > 0), 'nx should be an integer and >0.'
    assert(isinstance(ny, int) and ny > 0), 'ny should be an integer and >0.'
    assert(isinstance(nc, int) and nc > 0), 'nc should be an integer and >0.'
    self.nx = nx
    self.ny = ny
    self.nc = nc

    self.activation = activation
    if self.activation == 'relu':
      self.f = self._f_relu
      self.dfdx = self._dfdx_relu
    elif self.activation == 'sigmoid':
      self.f = self._f_sigmoid
      self.dfdx = self._dfdx_sigmoid
    else:
      assert False, "Error: activation function %s is not implemented." % \
        self.activation


  def forward_pass(self, x, save, w=None, b=None):
    """Perform forward pass on this layer using input x, return the output y.
    If save, then update self.x and self.y."""

    assert(w is None and b is None), \
      'w and b should be None, they are N/A for this layer.'
    assert(isinstance(x, np.ndarray)), 'x should be a numpy array.'
    assert(x.shape == (self.nc, self.nx, self.ny)), \
      'x should have the shape %r.' % x.shape
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
    return dLdx, None, None


  def _check_gradient_dLdx(self, dLdy, dLdx):

    x_dims = self.x.shape

    # g = numerical gradient
    g = np.zeros(x_dims)
    for ci, xi, yi in itertools.product(*[range(it) for it in x_dims]):
      x_eps = np.zeros(x_dims)
      x_eps[ci, xi, yi] = kEpsilonNumericalGrad
      dydx = (self.forward_pass(self.x + x_eps, save=False) - \
                  self.forward_pass(self.x - x_eps, save=False))  \
                  / 2.0 / kEpsilonNumericalGrad
      g[ci, xi, yi] = np.sum(dLdy * dydx)

    # Return check results
    res = np.max(np.square(g-dLdx)) < kAllowNumericalErr and \
      np.sum(g) != 0 and np.sum(dLdx) != 0
    if not res:
      print('Analytical dLdx = \n%r' % dLdx)
      print('Numerical dLdx = \n%r' % g)
      print('Max (analytical - numerical) error = %r' % np.max(dLdx - g))
      print('Min (analytical - numerical) error = %r' % np.min(dLdx - g))
      print('Error: %s check dLdx gradient failed.' % self.__class__)
    return res


  def _check_gradient_dLdw(self, dLdy, dLdw, dLdb):
    return True


  def _f_relu(self, x):
    """Implement ReLU function, return ReLU(x)."""
    return x * (x > 0)


  def _dfdx_relu(self, x):
    """Implement the gradient of ReLU function, return del ReLU(x)."""
    return (x > 0) * 1


  def _f_sigmoid(self, x):
    """Implement the sigmoid function, return sigmoid(x)."""
    return 1 / (1 + np.exp(-x))


  def _dfdx_sigmoid(self, x):
    """Implement the gradient of sigmoid function, return del sigmoid(x)."""
    f_x = self._f_sigmoid(x)
    return (1 - f_x)*f_x



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
    assert(isinstance(k, int) and k > 0 and isinstance(s, int) and s > 0)

    self.nx = nx
    self.ny = ny
    self.nc = nc
    self.k = k
    self.s = s

    self.nxo = int((nx - k)/s + 1)
    self.nyo = int((ny - k)/s + 1)

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
          sub_x = x[ci,
                    xi * self.s : xi * self.s + self.k,
                    yi * self.s : yi * self.s + self.k]
          y[ci, xi, yi] = self.f(sub_x)

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
    dLdx = np.zeros(self.x.shape)
    for ci in range(self.nc):
      for xii in range(self.nx):
        for yii in range(self.ny):
          for xoi in range(self.nxo):
            for yoi in range(self.nyo):
              if (xii - self.s * xoi < self.k) and \
                 (xii - self.s * xoi >= 0) and \
                 (yii - self.s * yoi < self.k) and \
                 (yii - self.s * yoi >= 0):
                sx = self.x[ci,
                            xoi * self.s : xoi * self.s + self.k,
                            yoi * self.s : yoi * self.s + self.k]
                dLdx[ci, xii, yii] += dLdy[ci, xoi, yoi] * \
                  self.dfdx(sx)[xii - xoi * self.s,
                                yii - yoi * self.s]

    # If save
    if save:
      self.dLdx = dLdx

    # Return
    return dLdx, None, None


  def _check_gradient_dLdx(self, dLdy, dLdx):

    x_dims = self.x.shape

    # g = numerical gradient
    g = np.zeros(x_dims)
    for ci, xi, yi in itertools.product(*[range(it) for it in x_dims]):
      x_eps = np.zeros(x_dims)
      x_eps[ci, xi, yi] = kEpsilonNumericalGrad
      dydx = (self.forward_pass(self.x + x_eps, save=False) - \
                  self.forward_pass(self.x - x_eps, save=False))  \
                  / 2.0 / kEpsilonNumericalGrad
      g[ci, xi, yi] = np.sum(dLdy * dydx)

    # Return check results
    res = np.max(np.square(g-dLdx)) < kAllowNumericalErr and \
      np.sum(g) != 0 and np.sum(dLdx) != 0
    if not res:
      print('Analytical dLdx = \n%r' % dLdx)
      print('Numerical dLdx = \n%r' % g)
      print('Max (analytical - numerical) error = %r' % np.max(dLdx - g))
      print('Min (analytical - numerical) error = %r' % np.min(dLdx - g))
      print('Error: %s check dLdx gradient failed.' % self.__class__)
    return res


  def _check_gradient_dLdw(self, dLdy, dLdw, dLdb):
    return True


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
    return np.ones(sx.shape) * (1/self.k/self.k)
