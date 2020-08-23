import numpy as np
from constants import *
from loss_functions import *


class SequentialNet:
    """
    A sequential neural net, i.e. any neural network that does not have recurrent
    connections, e.g. MLP, ConvNet
    """

    def __init__(self, input_dimensions, output_dimensions):

        assert(len(input_dimensions)==2 and \
               all([isinstance(it, int) and it>0 for it in input_dimensions])), \
               'Incorrect input_dimensions %r.' % input_dimensions
        assert(len(output_dimensions)==2 and \
               all([isinstance(it, int) and it>0 for it in output_dimensions])), \
               'Incorrect output_dimensions %r.' % output_dimensions

        # Layer 0 is the input layer, it has no computation, represented by None
        # here.
        self.layers = [None]

        self.nxi = input_dimensions[0]
        self.nyi = input_dimensions[1]
        self.nxo = output_dimensions[0]
        self.nyo = output_dimensions[1]

        self.layers_module = __import__('layers')
        self.layer_types = ['DenseLayer',
                            'ConvLayer',
                            'ActivationLayer',
                            'PoolingLayer']


    def reset_cache(self):
        """Reset cached temporary states for all layers."""
        for l in self.layers:
            l.reset_cache()


    def add_layer(self, layer_type, layer_params, w=None):
        """Augment the SequentialNet by a new layer, with layer_params depending
        on the layer_type. Optionally use an initial weight matrix w."""

        assert any([layer_type==it for it in self.layer_types]), \
            'Unknown layer_type %s' % layer_type

        self.layers.append(
            getattr(self.layers_module, layer_type)(*layer_params, w))


    def define_loss_function(self, loss_fxn_type):
        """Define the loss function object to evaluate the output y."""
        self.loss = LossFunction(loss_fxn_type)


    def normalize_data(self, x):
        """Normalize x, x can be a scalar, a vector, or a matrix."""
        norm_factor = np.abs(np.max(x))
        if norm_factor == 0:
            return x
        else:
            assert norm_factor > 0, 'norm_factor should not be <0.'
            return x * 1.0 / norm_factor


    def forward_pass(self, x0):
        """Perform forward pass using input x0, cache intermediate layers' states,
        return output xN."""
        x0 = self.normalize_data(x0)
        x = x0
        for l in self.layers:
            if l is not None:
                x = l.forward_pass(x, save=True)
        self.xn = x
        return self.xn


    def backprop(self, dLdxn):
        """Backpropagate the loss error dLdxn to update the SequentialNet."""
        self._backprop_to_layer(to_layer=0, save=True)


    def _backprop_to_layer(self, dLdxn, to_layer, save):
        """Backpropagate the loss error dLdxn up until to_layer, return dLdx,
        the loss gradient with respect to the input of that layer. If save,
        update the state of the SequentialNet as backprop happens."""
        for l in reversed(self.layers[to_layer:]):
            if l is not None:
                dLdx, dLdw, dLdb = l.backprop(dLdx, save)
        return dLdx, dLdw, dLdb


    def update_weights(self, batch_size):
        """Update the weights matrix, bias vectors, if any, at every layer."""
        for l in self.layers:
            if l is not None:
                l.update_weights(batch_size, l.dLdw=None, l.dLdb=None)


    def check_gradient_at_layer(self, i):
        """Check backprop gradient for the i-th layer using its locally stored
        dLdy."""
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

        x = self.layers[i].x
        w = self.layers[i].get_weights()
        w_dims = w.shape

        # Calculate gw, the numerical gradient for weights
        assert len(w_dims)==2, 'w_dim != 2 is not yet implemented.'
        gw = np.zeros(w_dims)
        for j in range(w_dims[0]):
            for k in range(w_dims[1]):
                w_nudge = np.zeros(w_dims)
                w_nudge[j][k] = kEpsilonNumericalGrad
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
                            / 2.0 / kEpsilonNumericalGrad

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
        x_dim = self.layers[i].get_input_size() + 1 # +1 for the bias term
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
            g[j] = (self.evaluate_loss(x_nudge_up, y) -
                    self.evaluate_loss(x_nudge_down, y)) \
                    / 2.0 / kEpsilonNumericalGrad
        # Discard the last element, which is the gradient on the bias term.
        g = g[:-1]

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
            'MLP.classify_one_hot returned an output that is not one hot {}'\
            .format(res)
        if y is not None:
            loss = self.evaluate_loss(xn, y)
        else:
            loss = None

        #print('loss = ', loss)
        #print('\n')
        return res, loss


    def print_weights(self, i):
        """Print weights for the i-th layer. 0-th is the input layer, -1 for the
        output layer."""
        if i == 0:
            print('Weights for the input layer: N/A.\n')
        else:
            print('MLP.print_weights() for the %d-th layer =\n' % i)
            print(self.layers[i].get_weights())
