import numpy as np
import matplotlib.pyplot as plt
import constants, utils
from layers import *
from loss_functions import *


class MLP:
  """
  A multi-layer perceptron.
  """
  # Definitions:

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


  def add_layer(self, n, activation, w=None, b=None):
    """Augment the MLP by a new layer of width n."""
    # Get the last layer's output dimension.
    if len(self.layers) == 0:
      in_dimension = self.nx0
    else:
      in_dimension = self.layers[-1].get_output_dims()[0]
    # Append the new layer.
    self.layers.append(DenseLayer(in_dimension, n, activation, w, b))


  def define_loss_function(self, loss_fxn_type):
    """Define the loss function object, to evaluate the output y."""
    self.loss = LossFunction(loss_fxn_type)


  def forward_pass(self, x0):
    """Perform forward pass using input x0, cache intermediate layer states,
    return output xN."""
    x0 = utils.normalize_data(x0)
    self.xn = self._forward_pass_plus(x0, save=True)
    return self.xn


  def _forward_pass_plus(self, x0, save):
    """Perform forward pass, with options to update the state of the MLP."""
    x = x0
    for l in self.layers:
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
    dLdx = dLdxn
    for l in reversed(self.layers[to_layer:]):
      # The last element of dLdx is the bias term, need not be propagated to the
      # previous layer
      dLdx, dLdw, dLdb = l.backprop(dLdx, save)

    return dLdx, dLdw, dLdb


  def update_weights(self, batch_size, eta):
    """Update the weights matrix in every layer, using the learning rate eta."""
    for l in self.layers:
      l.update_weights(batch_size, eta)


  def plot_gradient_distribution(self):
    """Plot the distribution of the values in dLdw in a histogram."""
    for l in self.layers:
      if l is not None:
        l.plot_gradient_distribution()
    plt.show(block=True)


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
    w = self.layers[i].w
    b = self.layers[i].b
    w_dims = w.shape

    # Calculate gw, the numerical gradient for weights
    assert len(w_dims)==2, 'w_dim != 2 is not yet implemented.'
    gw = np.zeros(w_dims)
    for j in range(w_dims[0]):
      for k in range(w_dims[1]):
        w_nudge = np.zeros(w_dims)
        w_nudge[j][k] = constants.kEpsilonNumericalGrad
        w_nudge_up = w + w_nudge
        w_nudge_down = w - w_nudge
        x_nudge_up = self.layers[i].forward_pass(
          x, save=False, w=w_nudge_up)
        x_nudge_down = self.layers[i].forward_pass(
          x, save=False, w=w_nudge_down)
        for l in self.layers[i+1::]:
          x_nudge_up = l.forward_pass(x_nudge_up, save=False)
          x_nudge_down = l.forward_pass(x_nudge_down, save=False)
        gw[j][k] = (self.evaluate_loss(x_nudge_up, y) - \
                    self.evaluate_loss(x_nudge_down, y)) \
                    / 2.0 / constants.kEpsilonNumericalGrad

    # Calculate dLdw, the analytical gradient from backprop
    dLdx, dLdw, dLdb = self._backprop_plus(dLdxn, save=False, to_layer=i)

    # Compare numerical and analytical gradients
    gradient_diff = gw - dLdw
    if np.all(abs(gradient_diff) < kAllowNumericalErr):
      return True
    else:
      print('MLP._check_gradient_from_layer_dLdw, numerical gradient =')
      print(gw)
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
    x_dim = self.layers[i].get_input_dims()[0] + 1 # +1 for the bias term
    x = self.layers[i].x

    # Calculate g, the numerical gradient
    g = np.zeros(x_dim)
    for j in range(x_dim):
      x_nudge = np.zeros(x_dim)
      x_nudge[j] = constants.kEpsilonNumericalGrad
      x_nudge_up = x + x_nudge
      x_nudge_down = x - x_nudge
      for l in self.layers[i::]:
        x_nudge_up = l.forward_pass(x_nudge_up, save=False)
        x_nudge_down = l.forward_pass(x_nudge_down, save=False)
      g[j] = (self.evaluate_loss(x_nudge_up, y) - self.evaluate_loss(x_nudge_down, y)) \
        / 2.0 / constants.kEpsilonNumericalGrad
    g = g[:-1] # Discard the last element, which is the gradient on the bias term.

    # Calculate dLdx, the analytical gradient from backprop
    dLdx, dLdw, dLdb = self._backprop_plus(dLdxn, save=False, to_layer=i)

    # Compare numerical and analytical gradients
    gradient_diff = g - dLdx
    if np.all(abs(gradient_diff) < kAllowNumericalErr):
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


  def get_layer_dims(self, i):
    """Get the width of the i-th layer. 0-th is the input layer, -1 for the
    output layer."""
    if i == 0:
      return (self.nx0,)
    else:
      return self.get_layer(i).get_output_dims()


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


  def summary(self):
    """Print a summary of the neural network."""
