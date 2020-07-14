import numpy as np

kEpsilonNumericalGrad = 1e-5


class Layer:
  """
  A single layer in an MLP, f(h) = f(w.dot(x))
  """
  # Definitions:

  # n = layer width
  # x = input vector
  # n_x = input vector length
  # w = weights matrix
  # h = intermediate variable, w.dot(x)
  # y = output vector

  # activation = activation function name
  # f = the activation function
  # dfdh = the derivative of the activation function

  # dLdx = backpropagation from dLdy to dLdx
  # dLdw = backpropagation from dLdy to dLdw

  def __init__(self, n_x, n, activation, w=None):

    assert(isinstance(n, int) and n > 0), 'n should be an integer and >0.'
    assert(isinstance(n_x, int) and n_x > 0), 'n_x should be an integer and >0.'

    self.n_x = n_x
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
      self.w = self._initialize_weights(self.n, self.n_x)
    else:
      assert(w.shape == (self.n, self.n_x)), \
        'User input w has the wrong dimensions {}'.format(w.shape)
      self.w = w


  def _initialize_weights(self, n, m):
    """Initialize a weights matrix of dimensions (n x m)."""
    return np.random.rand(self.n, self.n_x) * 2 - 1


  def forward_pass(self, x, save):
    """Perform forward pass on this layer using input x, return the output y.
    If save, update self.x, self.h, and self.y. """

    assert(isinstance(x, np.ndarray)), 'x should be a numpy array.'
    assert(len(x.shape)==1), 'x should be a vector.'

    h = self.normalize_vector(self.w.dot(x))
    y = self.f(h)
    if save:
      self.x = x
      self.h = h
      self.y = y

    # Return output of this layer
    return y


  def backprop(self, dLdy, save):
    """Calculate and return (dLdx, dLdw). If save, update w using dLdw, and save
    self.dLdy."""
    # dLdy = (n * 1)
    # dydh = dfdh = (n * n) diagonal
    # dhdx = (n * n_x) = w
    # dhdw = (n_x * 1) = x
    # dLdw = dydh * dLdy * dhdw.T
    # dLdx = dhdx.T * dydh * dLdy

    # Do backprop
    dydh = self.dfdh(self.h)
    dLdw = np.outer(dydh @ dLdy, self.x.T)
    dLdx = self.w.T @ dydh @ dLdy
    if save:
      self.dLdx = dLdx
      self.dLdw = dLdw
      self.w = self.w + self.dLdw
      self.dLdy = dLdy # Cache dLdy for gradient checking

    # Return dLdx for the next layer's backprop
    return dLdx, dLdw


  def check_gradient(self):
    """Assume we have performed backprop to update self.dLdx and self.dLdw,
    check them against numerical calculations from the same dLdy."""
    self._check_gradient_dLdx()
    # self._check_gradient_dLdw()


  def _check_gradient_dLdx(self):
    """Check input gradient dLdx against numerical calculation."""
    # dLdx = [L(x + eps) - L(x - eps)] / (2*eps)
    # dLdx = \sum_j dLdy_j * [y_j(x + eps) - y_j(x - eps)] / (2*eps)
    # g is the numerically calculated gradient, for comparison with self.dLdx
    g = np.zeros(self.n_x)
    for i in range(self.n_x):
      x_eps = np.zeros(self.n_x)
      x_eps[i] = kEpsilonNumericalGrad
      dydx = (self.forward_pass(self.x + x_eps, save=False) - \
              self.forward_pass(self.x - x_eps, save=False))  \
              / 2.0 / kEpsilonNumericalGrad
      g[i] = self.dLdy.dot(dydx)
    # print(self.dLdx)
    # print(g)
    # print(np.max(self.dLdx - g))
    # print(np.min(self.dLdx - g))


  def get_input_size(self):
    return self.n_x


  def get_width(self):
    return self.n


  def get_weights(self):
    return self.w


  def _f_sigmoid(self, h):
    """Evaluate the sigmoid function for h, where h is a vector."""
    assert(len(h.shape)==1), 'Input arg h should be a vector.'
    return 1 / (1 + np.exp(-h))


  def _dfdh_sigmoid(self, h):
    """Evaluate the gradient of sigmoid function at h, where h is a vector."""
    assert(len(h.shape)==1), 'Input arg h should be a vector.'
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
    return np.diag(f_h*(1 - f_h))


  def normalize_vector(self, h):
    """Normalize a vector h for numerical stability"""
    if abs(np.max(h)) != 0:
      return h / abs(np.max(h))
    else:
      return h


class LossFunction:
  """
  Loss function, specific implementations include (categorical) cross-entropy
  """
  # self.f = the loss function
  # self.dfdx = the gradient of the loss function at x

  def __init__(self, loss_fxn_type):
    if loss_fxn_type == "cce":
      self.f = self._f_cce
      self.dfdx = self._dfdx_cce
    else:
      assert(False), 'loss function %s is not implemented.' % loss_fxn_type


  def evaluate(self, x, y):
    return self.f(x, y)


  def get_gradient(self, x, y):
    return self.dfdx(x, y)


  def _f_cce(self, x, y):
    """Evaluate the categorical cross-entropy loss for x against the ground truth y."""
    # TODO: this implementation is for one-hot category only, add multi-class
    return -1*y.dot(np.log(x))


  def _dfdx_cce(self, x, y):
    """Evaluate the gradient of categorical cross-entropy loss at x"""
    # TODO: this implementation is for one-hot category only, add multi-class
    return x - y


class MLP:
  """
  A multi-layer perception
  """
  # Definitions:
  #
  # n_x0 = input data dimension
  # n_y = output and training data dimension
  # x0 = MLP input, (n_x0 * 1)
  # xn = MLP output, (n_y * 1). xn is the output of the n-th layer.
  #      The last layer output as the same dimensions as the training data
  # y = training data, (n_y * 1).
  # layers = a list of Layer objects, dynamically expand using add_layer()
  # loss = the loss function, used to evaluate the output xn

  def __init__(self, input_dimension, output_dimension):
    self.layers = []
    self.n_x0 = input_dimension
    self.n_y = output_dimension
    self.x0 = [0]*self.n_x0
    self.xn = [0]*self.n_y


  def add_layer(self, n, activation, w=None):
    """Augment the MLP by a new layer of width n."""
    # Get the last layer's output dimension.
    if len(self.layers) == 0:
      in_dimension = self.n_x0
    else:
      in_dimension = self.layers[-1].get_width()
    # Automatically extend n by +1 for bias term.
    in_dimension += 1
    # Append the new layer.
    self.layers.append(Layer(in_dimension, n, activation, w))


  def define_loss_function(self, loss_fxn_type):
    """Define the loss function object, to evaluate the output y."""
    self.loss = LossFunction(loss_fxn_type)


  def forward_pass(self, x0):
    """Perform forward pass using input x0, cache intermediate layer states,
    return output xN."""
    x = x0
    for l in self.layers:
      # Append +1 for the bias term. TODO: keep appending each time is not
      # efficient.
      x = np.append(x, 1)
      # Run forward pass
      x = l.forward_pass(x, save=True)

    # Save and return the last layer's output
    self.xn = x
    return self.xn


  def backprop(self, dLdxn):
    """Backpropagate loss error dLdxn to update the MLP."""
    dLdx = dLdxn
    for l in reversed(self.layers):
      dLdx, dLdw = l.backprop(dLdx, save=True)
      # The last element of dLdx is the bias term, need not be propagated to the
      # previous layer
      dLdx = dLdx[:-1]


  def check_gradient_at_layer(self, i):
    """Check backprop gradient for the i-th layer using its locally stored dLdy.
    """
    if i == 0:
      print('Gradient check is irrelevant for the input layer.')
    else:
      self.get_layer(i).check_gradient()


  def check_gradient_from_layer(self, i, y):
    """Check the whole MLP's dLdx and dLdw at layer i, i.e. how the MLP's loss
    changes as a function of changes in i-th layer's x and w. Where loss is
    evaluated against the ground truth y."""
    if i == 0:
      print('Irrelevant to check gradient against the input layer. Exit.')
    else:
      self._check_gradient_from_layer_dLdw(i, y)
      self._check_gradient_from_layer_dLdx(i, y)


  def _check_gradient_from_layer_dLdw(self, i, y):
    """Helper function for self.check_gradient_from_layer for dLdw."""
    w_dims = self.get_layer(i).get_weights().shape


  def _check_gradient_from_layer_dLdx(self, i, y):
    """Helper function for self.check_gradient_from_layer for dLdx."""
    if i > 0:
      i = i - 1
    x_dim = self.layers[i].get_input_size()
    x = self.layers[i].x

    # g is the numerically calculated gradient
    g = np.zeros(x_dim)

    for j in range(x_dim):
      x_nudge = np.zeros(x_dim)
      x_nudge[j] = kEpsilonNumericalGrad
      x_nudge_up = x + x_nudge
      x_nudge_down = x - x_nudge
      for l in self.layers[i::]:
        xn_nudge_up = l.forward_pass(x_nudge_up, save=False)
        xn_nudge_down = l.forward_pass(x_nudge_down, save=False)
      g[j] = (self.evaluate_loss(xn_nudge_up, y) - self.evaluate_loss(xn_nudge_down, y)) \
        / 2.0 / kEpsilonNumericalGrad

    print('MLP._check_gradient_from_layer_dLdx, numerically calculated gradient =')
    print(g)
    print('MLP._check_gradient_from_layer_dLdx, backpropagated gradient =')
    print(self.layers[i].dLdx)


  def calculate_loss_gradient(self, xn, y):
    """Calculate dL/dxn, gradient of loss at xn from the previous data sample,
    using training outcome y."""
    return self.loss.get_gradient(xn, y)


  def evaluate_loss(self, xn, y):
    """Evaluate the loss of a net output xn against the corresponding desired
    output y."""
    return self.loss.evaluate(xn, y)


  def get_layer(self, i):
    """Get the Layer object for the i-th layer. 0-th is the input layer, -1 is
    the output layer."""
    if i == 0:
      print('No Layer object for the input layer.')
      return None
    elif i < 0:
      return self.layers[i]
    else:
      return self.layers[i-1]


  def get_layer_width(self, i):
    """Get the width of the i-th layer. 0-th is the input layer, -1 for the
    output layer."""
    if i == 0:
      return self.n_x0
    else:
      return self.get_layer(i).get_width()


  def print_weights(self, i):
    """Print weights for the i-th layer. 0-th is the input layer, -1 for the
    output layer."""
    if i == 0:
      print('Weights for the input layer: N/A.\n')
    elif i == -1:
      print('MLP.print_weights() for the %d-th layer =\n' % i)
      print(self.layers[i].get_weights())
    else:
      print('MLP.print_weights() for the %d-th layer (i=%d) =\n' % \
            (len(self.layers)-1, i))
      print(self.layers[i-1].get_weights())


class NetTrainer:
  """
  Train a neural net.
  """

  def sgd(self, nn, x_train, y_train, epochs, batch_size, eta):
    """Train a neural net nn using batched stochastic gradient descent."""
    # eta is the learning rate.

    # Check input argument consistency
    assert(isinstance(nn, MLP)), \
      'Input neural net nn is not an instance of MLP class.'

    assert(x_train.shape[0] == y_train.shape[0]), \
      'x_train and y_train should have the same number of samples.'

    input_width = nn.get_layer_width(0)
    assert(x_train.shape[1] == input_width), \
      'x_train data has dimension %d, inconsistent with the neural net\'s input dimension %d.' \
      % (x_train.shape[1], input_width)

    output_width = nn.get_layer_width(-1)
    assert(y_train.shape[1] == output_width), \
      'y_train data has dimension %d, inconsistent with the neural net\'s output dimension %d.' \
      % (y_train.shape[1], output_width)

    num_samples = x_train.shape[0]
    assert(batch_size <= num_samples), 'batch_size [%d] > number of samples in x/y_train [%d].' \
      % (batch_size, num_samples)

    # Do batched sgd
    for i in range(epochs):
      cumulative_loss = 0
      cumulative_loss_gradient = [0]*output_width

      # Evaluate loss and loss gradient for a batch
      for j in range(batch_size):
        s = self._select_sample(j, num_samples)
        res = nn.forward_pass(x_train[s])
        cumulative_loss += nn.evaluate_loss(res, y_train[s])
        cumulative_loss_gradient += nn.calculate_loss_gradient(res, y_train[s])
        # print(res)
        # print(y_train[s])
        # print(nn.calculate_loss_gradient(res, y_train[s]))

      # Train for this epoch
      cumulative_loss = cumulative_loss / batch_size
      cumulative_loss_gradient = cumulative_loss_gradient / batch_size
      nn.backprop(cumulative_loss_gradient * eta)
      nn.check_gradient_at_layer(2)
      nn.check_gradient_from_layer(2, y_train[s])
      #nn.print_weights(1)
      #nn.print_weights(2)
      print('Epoch #%d: loss = %f\n' % (i, cumulative_loss))


  def _select_sample(self, count, num_samples):
    """Helper function to select sample from num_samples."""
    # Currently round-robin across all num_samples. Can also select at random.
    return count % num_samples
