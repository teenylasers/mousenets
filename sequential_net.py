import itertools
import numpy as np
import constants, utils
from loss_functions import *
from utils import *


###@@@ TODO
###@@@ SequentialNet only takes 2D images (multiple channels okay) as input and
###@@@ output. It cannot be used as it to construct MLP with 1D input/output.


class SequentialNet:
    """
    A sequential neural net, i.e. any neural network that does not have recurrent
    connections, e.g. MLP, ConvNet
    """

    def __init__(self): #, input_dimensions, output_dimensions):

        # assert(len(input_dimensions)==2 and \
        #        all([isinstance(it, int) and it>0 for it in input_dimensions])), \
        #        'Incorrect input_dimensions %r.' % input_dimensions
        # assert(len(output_dimensions)==2 and \
        #        all([isinstance(it, int) and it>0 for it in output_dimensions])), \
        #        'Incorrect output_dimensions %r.' % output_dimensions

        # Layer 0 is the input layer, it has no computation, represented by None
        # here.
        self.layers = [None]

        # self.nxi = input_dimensions[0]
        # self.nyi = input_dimensions[1]
        # self.nxo = output_dimensions[0]
        # self.nyo = output_dimensions[1]

        self.layers_module = __import__('layers')
        self.layer_types = ['DenseLayer',
                            'DenseLayer2D',
                            'ConvLayer',
                            'ActivationLayer',
                            'PoolingLayer']


    def reset_cache(self):
        """Reset cached temporary states for all layers."""
        for l in self.layers:
            if l is not None:
                l.reset_cache()


    def add_layer(self, layer_type, layer_params):
        """Augment the SequentialNet by a new layer, with layer_params depending
        on the layer_type. Optionally use an initial weight matrix w."""

        assert any([layer_type==it for it in self.layer_types]), \
            'Unknown layer_type %s' % layer_type

        self.layers.append(
            getattr(self.layers_module, layer_type)(*layer_params))


    def define_loss_function(self, loss_fxn_type):
        """Define the loss function object to evaluate the output y."""
        self.loss = LossFunction(loss_fxn_type)


    def forward_pass(self, x0):
        """Perform forward pass using input x0, cache intermediate layers' states,
        return output xN."""
        x0 = utils.normalize_data(x0)
        x = x0
        for l in self.layers:
            if l is not None:
                x = l.forward_pass(x, save=True)
        self.xn = x
        return self.xn


    def backprop(self, dLdxn):
        """Backpropagate the loss error dLdxn to update the SequentialNet."""
        self._backprop_to_layer(dLdxn, to_layer=0, save=True)


    def _backprop_to_layer(self, dLdxn, to_layer, save):
        """Backpropagate the loss error dLdxn up until to_layer, return dLdx,
        the loss gradient with respect to the input of that layer. If save,
        update the state of the SequentialNet as backprop happens."""
        dLdx = dLdxn
        for l in reversed(self.layers[to_layer:]):
            if l is not None:
                dLdx, dLdw, dLdb = l.backprop(dLdx, save)
        return dLdx, dLdw, dLdb


    def update_weights(self, batch_size, eta):
        """Update the weights matrix, bias vectors, if any, at every layer."""
        for l in self.layers:
            if l is not None:
                l.update_weights(batch_size, eta)


    def plot_gradient_distribution(self):
        for l in self.layers:
            if l is not None:
                l.plot_gradient_distribution()


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

        def _calculate_nudge_updn_output(w, w_nudge):
            """Helper function to calculate the net's output by nudging w up and
            down. Return (nudge_up_results, nudge_down_results)."""
            w_nudge_up = w + w_nudge
            w_nudge_down = w - w_nudge
            x_nudge_up = self.layers[i].forward_pass(
                x, save=False, w=w_nudge_up)
            x_nudge_down = self.layers[i].forward_pass(
                x, save=False, w=w_nudge_down)
            for l in self.layers[i+1::]:
                x_nudge_up = l.forward_pass(x_nudge_up, save=False)
                x_nudge_down = l.forward_pass(x_nudge_down, save=False)

            return x_nudge_up, x_nudge_down


        x = self.layers[i].x
        w = self.layers[i].w
        w_dims = w.shape

        # Calculate gw, the numerical gradient for weights
        gw = np.zeros(w_dims)
        if len(w_dims) == 2:
            for j, k in itertools.product(*[range(it) for it in w_dims]):
                w_nudge = np.zeros(w_dims)
                w_nudge[j][k] = constants.kEpsilonNumericalGrad
                x_nudge_up, x_nudge_down = _calculate_nudge_updn_output(w, w_nudge)
                gw[j][k] = (self.evaluate_loss(x_nudge_up, y) - \
                            self.evaluate_loss(x_nudge_down, y)) \
                            / 2.0 / constants.kEpsilonNumericalGrad
        elif len(w_dims) == 4:
            for j,k,l,m in itertools.product(*[range(it) for it in w_dims]):
                w_nudge = np.zeros(w_dims)
                w_nudge[j][k][l][m] = constants.kEpsilonNumericalGrad
                x_nudge_up, x_nudge_down = _calculate_nudge_updn_output(w, w_nudge)
                gw[j][k][l][m] = (self.evaluate_loss(x_nudge_up, y) - \
                            self.evaluate_loss(x_nudge_down, y)) \
                            / 2.0 / constants.kEpsilonNumericalGrad
        else:
            assert False, 'Unexpected len(w_dims). Received w_dims = {} for {}'\
                .format(w_dims, self.layers[i].__class__)


        # Calculate dLdw, the analytical gradient from backprop
        dLdx, dLdw, dLdb = self._backprop_to_layer(dLdxn, to_layer=i, save=False)

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

        def _calculate_nudge_updn_output(x, x_nudge):
            """Helper function to calculate the net's output by nudging x up and
            down. Return (nudge_up_results, nudge_down_results)."""
            x_nudge_up = x + x_nudge
            x_nudge_down = x - x_nudge
            for l in self.layers[i::]:
                x_nudge_up = l.forward_pass(x_nudge_up, save=False)
                x_nudge_down = l.forward_pass(x_nudge_down, save=False)

            return x_nudge_up, x_nudge_down


        x = self.layers[i].x
        x_dims = x.shape

        # Calculate g, the numerical gradient
        g = np.zeros(x_dims)
        if len(x_dims) == 1:
            for j in range(x_dims):
                x_nudge = np.zeros(x_dims)
                x_nudge[j] = constants.kEpsilonNumericalGrad
                x_nudge_up, x_nudge_down = _calculate_nudge_updn_output(x, x_nudge)
                g[j] = (self.evaluate_loss(x_nudge_up, y) -
                        self.evaluate_loss(x_nudge_down, y)) \
                        / 2.0 / constants.kEpsilonNumericalGrad
                # Discard the last element, which is the gradient on the bias term.
                #g = g[:-1]
        elif len(x_dims) == 3:
            for j,k,l in itertools.product(*[range(it) for it in x_dims]):
                x_nudge = np.zeros(x_dims)
                x_nudge[j][k][l] = constants.kEpsilonNumericalGrad
                x_nudge_up, x_nudge_down = _calculate_nudge_updn_output(x, x_nudge)
                g[j][k][l] = (self.evaluate_loss(x_nudge_up, y) -
                              self.evaluate_loss(x_nudge_down, y)) \
                              / 2.0 / constants.kEpsilonNumericalGrad
        else:
            assert False, 'Unexpected len(x_dims). Received x_dims = {} for {}'\
                .format(x_dims, self.layers[i].__class__)


        # Calculate dLdx, the analytical gradient from backprop
        dLdx, dLdw, dLdb = self._backprop_to_layer(dLdxn, to_layer=i, save=False)

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


    def get_layer(self, i):
        """Get the Layer object for the i-th layer. 0-th is the input layer, -1 is
        the output layer."""
        if i == 0:
            print('No Layer object for the input layer.')
            return None
        else:
            return self.layers[i]


    def get_layer_dims(self, i):
        """Get the width of the i-th layer. 0-th is the input layer, -1 for the
        output layer."""
        if i == 0:
            return self.layers[1].get_input_dims()
        else:
            return self.get_layer(i).get_output_dims()


    def print_weights(self, i):
        """Print weights for the i-th layer. 0-th is the input layer, -1 for the
        output layer."""
        if i == 0:
            print('Weights for the input layer: N/A.\n')
        else:
            print('MLP.print_weights() for the %d-th layer =\n' % i)
            print(self.layers[i].get_weights())


    def get_num_activation_layers(self):
        """Return the number of activation layers in this net."""
        n = 0
        for l in self.layers:
            if l is not None and l.HAS_ACTIVATION:
                n = n + 1
        return n
