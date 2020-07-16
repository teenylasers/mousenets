import numpy as np


kEpsilonNumericalGrad = 1e-5
kAllowNumErr = 1e-4


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
    return np.random.rand(self.n, self.nx) * 2 - 1


  def reset_cache(self):
    """Clear cached states. self.x, h, dLdx, dLdw, and y."""
    self.x = None
    self.h = None
    self.y = None
    self.dLdx = None
    # For batch training, we accumulate dLdw from each sample and update w at
    # the end of the batch.
    self.dLdw = []


  def forward_pass(self, x, save, w=None):
    """Perform forward pass on this layer using input x, return the output y.
    If cache, then append this state to self.x, self.h, and self.y. """

    assert(isinstance(x, np.ndarray)), 'x should be a numpy array.'
    assert(len(x.shape)==1), 'x should be a vector.'
    if w is not None:
      assert(not save), \
        'When user specifies a weights matrix w, cannot save the results to '\
        'modify the layer.'

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
      self.dLdw.append(dLdw)
      self.dLdy = dLdy # Cache dLdy for gradient checking

    # Return dLdx for the next layer's backprop
    return dLdx, dLdw


  def update_weights(self, dLdw):
    """Given a 3D matrix dLdw, update self.w. Matrix dLdw matrix size is
    (self.n * self.nx * batch_size)"""
    # Average across dLdw from multiple samples.
    batch_size = len(dLdw) * 1.0
    dw = sum(np.array(dLdw)) / batch_size
    # Adjust self.w
    self.w = self.w - dw


  def check_gradient(self):
    """Assume we have performed backprop to update self.dLdx and self.dLdw,
    check them against numerical calculations from the same dLdy."""
    self._check_gradient_dLdx()
    # self._check_gradient_dLdw()


  def _check_gradient_dLdx(self):
    """Check input gradient dLdx against numerical calculation."""
    # dLdx = [L(x + eps) - L(x - eps)] / (2*eps)
    # dLdx = \sum_j dLdy_j * [y_j(x + eps) - y_j(x - eps)] / (2*eps)

    # g = numerical gradient
    g = np.zeros(self.nx)
    for i in range(self.nx):
      x_eps = np.zeros(self.nx)
      x_eps[i] = kEpsilonNumericalGrad
      dydx = (self.forward_pass(self.x + x_eps, save=False) - \
              self.forward_pass(self.x - x_eps, save=False))  \
              / 2.0 / kEpsilonNumericalGrad
      g[i] = self.dLdy.dot(dydx)

    # dLdx, dLdw = analytical gradient
    dLdx, dLdw = backprop(self.dLdy, save=False)

    # print output. TODO: check and return bool
    print(dLdx)
    print(g)
    print(np.max(dLdx - g))
    print(np.min(dLdx - g))


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



class LossFunction:
  """
  Loss function, specific implementations include (categorical) cross-entropy
  """
  # self.f = the loss function
  # self.dfdx = the gradient of the loss function at x

  def __init__(self, loss_fxn_type):
    if loss_fxn_type == 'cce' or loss_fxn_type == 'ce':
      self.f = self._f_ce
      self.dfdx = self._dfdx_ce
    else:
      assert(False), 'loss function %s is not implemented.' % loss_fxn_type


  def evaluate(self, x, y):
    return self.f(x, y)


  def get_gradient(self, x, y):
    return self.dfdx(x, y)


  def _f_ce(self, x, y):
    """Evaluate the cross-entropy loss for x against the ground truth y."""
    return -1*y.dot(np.log(x))


  def _dfdx_ce(self, x, y):
    """Evaluate the gradient of cross-entropy loss at x"""
    return -y / x


class MLP:
  """
  A multi-layer perception
  """
  # Definitions:
  #
  # nx0 = input data dimension
  # ny = output and training data dimension
  # x0 = MLP input, (nx0 * 1)
  # xn = MLP output, (ny * 1). xn is the output of the n-th layer.
  #      The last layer output as the same dimensions as the training data
  # y = training data, (ny * 1).
  # layers = a list of Layer objects, dynamically expand using add_layer()
  # loss = the loss function, used to evaluate the output xn

  def __init__(self, input_dimension, output_dimension):
    self.layers = []
    self.nx0 = input_dimension
    self.ny = output_dimension
    self.x0 = [0]*self.nx0
    self.xn = [0]*self.ny


  def reset_cache(self):
    """Reset cached temporary states for all layers."""
    for l in self.layers:
      l.reset_cache()


  def add_layer(self, n, activation, w=None):
    """Augment the MLP by a new layer of width n."""
    # Get the last layer's output dimension.
    if len(self.layers) == 0:
      in_dimension = self.nx0
    else:
      in_dimension = self.layers[-1].get_width()
    # Automatically extend n by +1 for bias term.
    in_dimension += 1
    # Append the new layer.
    self.layers.append(Layer(in_dimension, n, activation, w))


  def define_loss_function(self, loss_fxn_type):
    """Define the loss function object, to evaluate the output y."""
    self.loss = LossFunction(loss_fxn_type)


  def normalize_data(self, v):
    """Normalize the data vector v."""
    norm_factor = np.abs(np.max(v))
    if norm_factor != 0:
      return v * 1.0 / norm_factor
    else:
      return v


  def forward_pass(self, x0):
    """Perform forward pass using input x0, cache intermediate layer states,
    return output xN."""
    x0 = self.normalize_data(x0)
    self.xn = self._forward_pass_plus(x0, save=True)
    return self.xn


  def _forward_pass_plus(self, x0, save):
    """Perform forward pass, with options to update the state of the MLP."""
    x = x0
    for l in self.layers:
      # Append +1 for the bias term.
      x = np.append(x, 1)
      # Run forward pass
      x = l.forward_pass(x, save)

    # Return the last layer's output
    return x


  def backprop(self, dLdxn):
    """Backpropagate loss error dLdxn to update the MLP."""
    self._backprop_plus(dLdxn, save=True, to_layer=0)


  def _backprop_plus(self, dLdxn, save, to_layer):
    """Backpropagate loss error dLdxn, with options to update the state of the
    MLP or up until a certain layer in self.layers, return dLdx, the loss
    gradient with respect to the input to that layer."""
    # Append a 1, because the for-loop expects every layer's dLdx to contain a
    # bias term as the last element.
    dLdx = np.append(dLdxn, 1)
    for l in reversed(self.layers[to_layer:]):
      # The last element of dLdx is the bias term, need not be propagated to the
      # previous layer
      dLdx, dLdw = l.backprop(dLdx[:-1], save)

    return dLdx, dLdw


  def update_weights(self):
    """Update the weights matrix in every layer."""
    for l in self.layers:
      l.update_weights(l.dLdw)


  def check_gradient_at_layer(self, i):
    """Check backprop gradient for the i-th layer using its locally stored dLdy.
    """
    if i == 0:
      print('Gradient check is irrelevant for the input layer.')
    else:
      self.get_layer(i).check_gradient()


  def check_gradient_from_layer(self, i, y, dLdxn):
    """Check the whole MLP's dLdx and dLdw at layer i, i.e. how the MLP's loss
    changes as a function of changes in i-th layer's x and w. Where loss is
    evaluated against the ground truth y."""
    if i == 0:
      print('Irrelevant to check gradient against the input layer. Exit.')
      return True
    else:
      return self._check_gradient_from_layer_dLdw(i, y, dLdxn) and \
        self._check_gradient_from_layer_dLdx(i, y, dLdxn)


  def _check_gradient_from_layer_dLdw(self, i, y, dLdxn):
    """Helper function for self.check_gradient_from_layer for dLdw."""
    # Convert i to self.layers index
    if i > 0:
      i = i - 1
    else:
      assert False, 'Cannot calculate dLdw for layer number %d.' % i
    x = self.layers[i].x
    w = self.layers[i].get_weights()
    w_dims = w.shape

    # Calculate g, the numerical gradient
    g = np.zeros(w_dims)
    for j in range(w_dims[0]):
      for k in range(w_dims[1]):
        w_nudge = np.zeros(w_dims)
        w_nudge[j][k] = kEpsilonNumericalGrad
        w_nudge_up = w + w_nudge
        w_nudge_down = w - w_nudge
        x_nudge_up = self.layers[i].forward_pass(x, save=False, w=w_nudge_up)
        x_nudge_down = self.layers[i].forward_pass(x, save=False, w=w_nudge_down)
        for l in self.layers[i+1::]:
          x_nudge_up = l.forward_pass(x_nudge_up, save=False)
          x_nudge_down = l.forward_pass(x_nudge_down, save=False)
        g[j][k] = (self.evaluate_loss(x_nudge_up, y) - self.evaluate_loss(x_nudge_down, y)) \
          / 2.0 / kEpsilonNumericalGrad

    # Calculate dLdw, the analytical gradient from backprop
    dLdx, dLdw = self._backprop_plus(dLdxn, save=False, to_layer=i)

    # Compare numerical and analytical gradients
    gradient_diff = g - dLdw
    if np.all(abs(gradient_diff) < kAllowNumErr):
      return True
    else:
      print('MLP._check_gradient_from_layer_dLdw, numerical gradient =')
      print(g)
      print('MLP._check_gradient_from_layer_dLdw, backpropagated gradient =')
      print(dLdw)
      return False


  def _check_gradient_from_layer_dLdx(self, i, y, dLdxn):
    """Helper function for self.check_gradient_from_layer for dLdx."""
    # Convert i to self.layers index
    if i > 0:
      i = i - 1
    else:
      assert False, 'Cannot calculate dLdx for layer number %d.' % i
    x_dim = self.layers[i].get_input_size()
    x = self.layers[i].x

    # Calculate g, the numerical gradient
    g = np.zeros(x_dim)
    for j in range(x_dim):
      x_nudge = np.zeros(x_dim)
      x_nudge[j] = kEpsilonNumericalGrad
      x_nudge_up = x + x_nudge
      x_nudge_down = x - x_nudge
      for l in self.layers[i::]:
        x_nudge_up = l.forward_pass(x_nudge_up, save=False)
        x_nudge_down = l.forward_pass(x_nudge_down, save=False)
      g[j] = (self.evaluate_loss(x_nudge_up, y) - self.evaluate_loss(x_nudge_down, y)) \
        / 2.0 / kEpsilonNumericalGrad

    # Calculate dLdx, the analytical gradient from backprop
    dLdx, dLdw = self._backprop_plus(dLdxn, save=False, to_layer=i)

    # Compare numerical and analytical gradients
    gradient_diff = g - dLdx
    if np.all(abs(gradient_diff) < kAllowNumErr):
      return True
    else:
      print('MLP._check_gradient_from_layer_dLdx, numerical gradient =')
      print(g)
      print('MLP._check_gradient_from_layer_dLdx, backpropagated gradient =')
      print(dLdx)
      return False


  def calculate_loss_gradient(self, xn, y):
    """Calculate dL/dxn, gradient of loss at xn from the previous data sample,
    using training outcome y."""
    return self.loss.get_gradient(xn, y)


  def evaluate_loss(self, xn, y):
    """Evaluate the loss of a net output xn against the corresponding desired
    output y."""
    return self.loss.evaluate(xn, y)


  def classify_one_hot(self, x0, y=None):
    """For a test input x0, return the neural net's classification. If the
    expected output y is provided, calculate and return the loss."""
    xn = self.forward_pass(x0)
    xn_max = np.max(xn)
    res = (xn == xn_max).astype(int)
    #print('xn = ', xn)
    #print('res = ', res)
    #print('y = ', y)
    assert(np.sum(res)==1), \
      'MLP.classify_one_hot returned an output that is not one hot {}'.format(res)
    if y is not None:
      loss = self.evaluate_loss(xn, y)
    else:
      loss = None
    #print('loss = ', loss)
    #print('\n')
    return res, loss


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
      return self.nx0
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

  def sgd(self, nn, x_train, y_train, epochs, batch_size, eta=None):
    """Train a neural net nn using batched stochastic gradient descent, return
    the trained neural net."""
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

    # Initialize loss_history
    loss_history = []

    # Initialize learning rate
    eta = self._get_etas(epochs, eta)

    # Do batched sgd
    for i in range(epochs):
      nn.reset_cache()
      cumulative_loss = 0
      cumulative_loss_gradient = np.zeros(output_width)
      loss_grads = np.zeros((batch_size, output_width))

      # Evaluate loss and loss gradient for a batch
      for j in range(batch_size):
        s = self._select_sample(j, num_samples)
        res = nn.forward_pass(x_train[s])
        cumulative_loss += nn.evaluate_loss(res, y_train[s])
        loss_grad = nn.calculate_loss_gradient(res, y_train[s])
        nn.backprop(loss_grad * eta)

      # Train for this epoch
      cumulative_loss = cumulative_loss / batch_size
      nn.update_weights()
      #weights_before = nn.get_layer(1).get_weights()
      #weights_after = nn.get_layer(1).get_weights()
      #delta_w = weights_after - weights_before
      #plt.imshow(delta_w)
      #plt.show()

      #print('Epoch #%d: loss = %f\n' % (i, cumulative_loss))
      loss_history.append(cumulative_loss)

    return nn, loss_history


  def _select_sample(self, count, num_samples):
    """Helper function to select sample from num_samples."""
    return np.random.randint(low=0, high=num_samples-1)


  def _get_etas(self, epochs, eta):
    assert(epochs>0), 'num epochs must be >0.'
    if eta is None:
      eta_init = 8
      eta_prelim = [eta_init**(2**-n) for n in range(epochs)]
      return [it if it > 1 else 1 for it in eta_prelim]
    else:
      return [eta]*epochs


  def evaluate(self, nn, x_test, y_test):
    """Evaluate the neural network nn against the test data set."""
    assert(x_test.shape[0] == y_test.shape[0]), \
      'x_test.shape = {}, y_test.shape = {}, inconsistent sizes.'\
      .format(x_test.shape, y_test.shape)
    num_tests = x_test.shape[0]
    num_correct = 0
    cumulative_loss = 0

    # Check test samples
    for i in range(num_tests):
      res, loss = nn.classify_one_hot(x_test[i], y_test[i])
      cumulative_loss += loss
      if self._compare_results(res, y_test[i]):
        num_correct += 1

    # Compile total stats
    cumulative_loss = cumulative_loss / num_tests
    accuracy = num_correct * 1.0 / num_tests
    return accuracy, cumulative_loss


  def _compare_results(self, y0, y1):
    """Helper function to compare 2 results, ignoring any formatting or data
    type differences"""
    # Currently only implemented for one-hot encoded classification output.
    # Augment as needed.
    y0 = y0.astype(int)
    y1 = y1.astype(int)
    return np.sum(y0)==np.sum(y1) and np.sum(y0) == 1 and \
      np.where(y0==1) == np.where(y1==1)
