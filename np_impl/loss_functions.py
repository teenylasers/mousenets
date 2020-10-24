import numpy as np


class LossFunction:
  """
  Loss function, specific implementations include (categorical) cross-entropy
  """
  # self.f = the loss function
  # self.dfdx = the gradient of the loss function at x

  def __init__(self, loss_fxn_type):
    if loss_fxn_type == 'cce' or loss_fxn_type == 'ce':
      # (Categorical) cross-entropy
      self.f = self._f_ce
      self.dfdx = self._dfdx_ce
    elif loss_fxn_type == 'sce':
      # Softmax-(C)CE combined
      self.f = self._f_sce
      self.dfdx = self._dfdx_sce
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
    """Evaluate the gradient of cross-entropy loss at x."""
    return -y / x


  def _f_sce(self, x, y):
    """Evaluate the softmax-cross-entropy loss for x against the ground truth y."""
    return np.log(np.sum(np.exp(x))) - y.dot(x)


  def _dfdx_sce(self, x, y):
    """Evaluate the gradient of softmax-cross-entropy loss at x."""

    def softmax(x):
      exp_x = np.exp(x)
      assert np.sum(exp_x) != 0
      return exp_x / np.sum(exp_x)

    return softmax(x) - y
