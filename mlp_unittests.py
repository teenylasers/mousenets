import unittest
import numpy as np

from mlp import *

kAllowNumErr = 1e-4

class MLPTest(unittest.TestCase):

    def _check_sigmoid(self, x, w, y):
        h = w.dot(np.append(x, 1))
        h = h/abs(np.max(h))
        x1 = 1/(1 + np.exp(-h))
        if (all([it < kAllowNumErr for it in (x1-y)])):
            return True
        else:
            print('Sigmoid function implementation is incorrect: \n'
                  'should be {} \ninstead got {}'.format(x1, y))
            return False

    def test_gradient(self):
        """Check analytical gradient from backprop against numerically
        calculated gradient."""
        # Set up a 2-layer MLP of width 2.
        n0 = 2
        n1 = 2
        mlp = MLP(input_dimension = n0, output_dimension = n1)
        w = np.arange(6).reshape(2,3) * 0.1
        mlp.add_layer(n1, 'sigmoid', w)
        mlp.define_loss_function('cce')
        x0 = np.array([1.5, 2.5])
        y = np.array([1, 0])

        # Forward pass
        x_out = mlp.forward_pass(x0)

        # Check sigmoid
        assert(self._check_sigmoid(x0, mlp.get_layer(1).get_weights(), x_out))

        # Run backprop once
        loss = mlp.evaluate_loss(x_out, y)
        loss_grad = mlp.calculate_loss_gradient(x_out, y)
        print('loss = %f' % loss)
        print('loss_grad =', loss_grad)
        print('w = ', mlp.get_layer(1).get_weights())
        mlp.backprop(loss_grad)
        print('w = ', mlp.get_layer(1).get_weights())

        # Check gradient
        mlp.check_gradient_from_layer(1, y)



if __name__ == '__main__':
    unittest.main()
