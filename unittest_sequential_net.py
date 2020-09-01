import random, unittest
import numpy as np
from constants import *
from mlp import *
from sequential_net import *


class MLPTest(unittest.TestCase):
    """
    Test a simple MLP implemented using SequentialNet.
    """
    def _construct_mlp(self, activation_function):
        """Construct and return an MLP using MLP with layer_width and randomly
        generated weights. All layers except the last uses activation_function,
        the last layer uses softmax. Use cce loss function."""

        layer_widths = self._generate_mlp_params()

        input_dimension = layer_widths[0]
        output_dimension = layer_widths[-1]
        mlp = MLP(input_dimension, output_dimension)
        for l in layer_widths[1:-1]:
            mlp.add_layer(l, activation_function)
        mlp.add_layer(layer_widths[-1], 'softmax')
        mlp.define_loss_function('cce')
        return mlp, input_dimension, output_dimension


    def _generate_mlp_params(self):
        """Generate a random set of params for MLP."""
        layer_widths = [random.choice(range(9,36))]
        num_layers = random.choice([2,3,4])
        for l in range(num_layers):
            if layer_widths[-1] == 1:
                continue
            else:
                layer_widths.append(random.choice(range(1,layer_widths[-1])))
        return layer_widths


    def test_mlp_gradient_check(self):

        test_activations = ['relu', 'sigmoid', 'softmax']
        num_tests = 3

        for activation in test_activations:
            for i in range(num_tests):

                print('MLP with %s, test %d: ' % (activation, i))
                mlp, input_dimension, output_dimension = \
                    self._construct_mlp(activation)
                x0 = np.random.rand(input_dimension) * 234.6
                xn = mlp.forward_pass(x0)
                y = np.zeros(output_dimension)
                y[random.choice(range(output_dimension))] = 1

                loss = mlp.evaluate_loss(xn, y)
                loss_grad = mlp.calculate_loss_gradient(xn, y)

                assert(mlp.check_gradient_from_layer(1, y, loss_grad))



class ConvNetTest(unittest.TestCase):


    def _construct_convnet(self, num_output, activation_fxn, pooling_fxn):
        """Construct and return a ConvNet with 1 ConvLayer, 1 ActivationLayer,
        1 PoolingLayer, and 1 FC-Layer implemented using DenseLayer2D. Use sce
        loss function."""

        cn = SequentialNet()

        # Layer 1: ConvLayer
        conv_params = self._generate_convlayer_params()
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


    def _generate_convlayer_params(self):
        """Generate a random set of params for a ConvLayer."""
        nx = random.choice([2,5,8])
        ny = random.choice([2,5,8])
        nc = 1
        k = 1 if min(nx,ny)==1 else random.choice(range(1,min(nx, ny)))
        c = random.choice([3,4,5,7])
        s = random.choice([1,2])
        p = random.choice([0,1,2])
        return (nx, ny, nc, k, c, s, p)


    def test_convnet_gradient_check(self):

        num_output = random.choice([2,3,4,5])
        num_tests = 5

        for i in range(num_tests):

            print('Test ConvNet %d: ' % i)

            convnet = self._construct_convnet(
                num_output, activation_fxn='relu', pooling_fxn='average')
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
                    print('{}, input shape {}'.format(l.__class__, l.x.shape))
            assert(convnet.check_gradient_from_layer(1, y, loss_grad))



if __name__ == '__main__':
    unittest.main()
