import random, unittest
import numpy as np
from constants import *
from sequential_net import *


def construct_convnet(num_output, activation_fxn, pooling_fxn):
    """Construct and return a ConvNet with 1 ConvLayer, 1 ActivationLayer,
    1 PoolingLayer, and 1 FC-Layer implemented using DenseLayer2D. Use sce
    loss function."""

    cn = SequentialNet()

    # Layer 1: ConvLayer
    conv_params = _generate_convlayer_params()
    print('ConvLayer: nx = %d, ny = %d, nc = %d, k = %d, c = %d, s = %d, '
          'p = %d.' % (conv_params[0], conv_params[1], conv_params[2],
                       conv_params[3], conv_params[4], conv_params[5],
                       conv_params[6]))
    cn.add_layer('ConvLayer', conv_params)

    # Layer 2: ActivationLayer
    nx = cn.get_layer(-1).nxo
    ny = cn.get_layer(-1).nyo
    nc = cn.get_layer(-1).c
    print('ActivationLayer: nx = %d, ny = %d, nc = %d.' % (nx, ny, nc))
    cn.add_layer('ActivationLayer', (nx, ny, nc, activation_fxn))

    # Layer 3: PoolingLayer
    k = 1 if nx == 1 or ny == 1 else random.choice(range(1,min(nx, ny)))
    s = 1 if k == 1 else random.choice(range(1,k))
    print('PoolingLayer: nx = %d, ny = %d, k = %d, s = %d.' % \
          (nx, ny, k, s))
    cn.add_layer('PoolingLayer', (nx, ny, nc, k, s, pooling_fxn))

    # Layer 4: FC-Layer
    nx = cn.get_layer(-1).nxo
    ny = cn.get_layer(-1).nyo
    nc = cn.get_layer(-1).nc
    print('FC-layer: nx = %d, ny = %d, nc = %d.' % (nx, ny, nc))
    cn.add_layer('DenseLayer2D', (nx, ny, nc, num_output, activation_fxn))

    # Softmax + cross-entropy loss function
    cn.define_loss_function('sce')

    # Return the ConvNet
    return cn


def _generate_convlayer_params():
    """Generate a random set of params for a ConvLayer."""
    nx = random.choice([2,5,8])
    ny = random.choice([2,5,8])
    nc = 1
    k = 1 if min(nx,ny)==1 else random.choice(range(1,min(nx, ny)))
    c = random.choice([3,4,5,7])
    s = random.choice([1,2])
    p = random.choice([0,1,2])
    return (nx, ny, nc, k, c, s, p)



class ConvNetTest(unittest.TestCase):
    """
    Construct a convolutional neural net using SequentialNet. Test forward_pass
    and backpropagation, do gradient check.
    """

    def test_convnet_test(self):

        cn = SequentialNet()
        # Layer 1: ConvLayer
        conv1_params = (4, 4, 1, 4, 6, 1, 0) # (nx, ny, nc, k, c, s, p)
        cn.add_layer('ConvLayer', conv1_params)
        # Layer 2: ActivationLayer
        act1_params = (1, 1, 6, 'sigmoid') # (nx, ny, nc, activation_fxn)
        cn.add_layer('ActivationLayer', act1_params)
        # Layer 3: ConvLayer
        conv2_params = (1, 1, 6, 1, 2, 1, 0) # (nx, ny, nc, k, c, s, p)
        cn.add_layer('ConvLayer', conv2_params)
        # Softmax + CCE loss function
        cn.define_loss_function('sce')

        num_output = 2
        x0 = np.random.rand(
            cn.get_layer(1).nc,
            cn.get_layer(1).nx,
            cn.get_layer(1).ny) * 10
        xn = cn.forward_pass(x0)
        y = np.zeros(num_output)
        y[random.choice(range(num_output))] = 1

        loss = cn.evaluate_loss(xn, y)
        loss_grad = cn.calculate_loss_gradient(xn, y)

        print(loss)


    def test_convnet_gradient_check(self):

        return

        num_output = random.choice([2,3,4,5])
        num_tests = 3
        test_activations = ['relu', 'sigmoid']
        test_pooling = ['average', 'max']

        for activation_fxn in test_activations:
            for pooling_fxn in test_pooling:
                for i in range(num_tests):

                    print('Test ConvNet %d: with %s and %s-pooling' % \
                          (i, activation_fxn, pooling_fxn))

                    convnet = construct_convnet(
                        num_output, activation_fxn, pooling_fxn)
                    x0 = np.random.rand(
                        convnet.get_layer(1).nc,
                        convnet.get_layer(1).nx,
                        convnet.get_layer(1).ny) * 10
                    xn = convnet.forward_pass(x0)
                    y = np.zeros(num_output)
                    y[random.choice(range(num_output))] = 1

                    loss = convnet.evaluate_loss(xn, y)
                    loss_grad = convnet.calculate_loss_gradient(xn, y)

                    for l in convnet.layers:
            			      if l is not None:
            			          print('{}, input shape {}'.format(
                                l.__class__, l.x.shape))

                    assert(convnet.check_gradient_from_layer(1, y, loss_grad))



if __name__ == '__main__':
    unittest.main()
