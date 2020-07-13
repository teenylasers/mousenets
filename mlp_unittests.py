import unittest
import numpy as np

from mlp import *

class MLPTest(unittest.TestCase):

    def testGradient(self):
        """Check analytical gradient from backprop against numerically
        calculated gradient."""
        # Set up a 2-layer MLP of width 2.
        n_l0 = 2
        n_l1 = 2
        mlp = MLP(input_dimension = n_l0, output_dimension = n_l1)
        mlp.add_layer(n_l1, 'sigmoid')
        mlp.define_loss_function('cce')

        # Test data
        x0 = np.array([1.5, 2.5])
        w = mlp.get_layer(1).get_weights()
        h = w.dot(np.append(x0, 1))
        h = h/abs(np.max(h))
        x1 = 1/(1 + np.exp(-h))
        y = np.array([1, 0])

        # Forward pass
        x_out = mlp.forward_pass(x0)

        loss = mlp.evaluate_loss(x_out, y)
        loss_grad = mlp.calculate_loss_gradient(x_out, y)
        print('x1 =', x1)
        print('x_out =', x_out)
        print('loss = %f' % loss)
        print('loss_grad =', loss_grad)

        # Backprop
        print('w = ', mlp.get_layer(1).get_weights())
        mlp.backprop(loss_grad)
        print('w = ', mlp.get_layer(1).get_weights())
        mlp.check_gradient_from_layer(1, y)



if __name__ == '__main__':
    unittest.main()
