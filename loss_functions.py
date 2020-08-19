import numpy as np


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
