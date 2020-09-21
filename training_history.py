import numpy as np
import matplotlib.pyplot as plt


# Data structures to represent neural net training history.



class FrozenObject:
    """
    An object where you cannot add class attributes after initialization.
    """
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class, cannot add class attributes"
                            % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True



class ActivationValuesHistory(FrozenObject):
    """
    Structure to represent the history of activation values evolution over
    epochs.
    """
    # We do not check input argument validity here, because we expect this
    # class to be used internally, and the checking has been done by the
    # internal user of this class.
    def __init__(self, num_epochs):
        self.mean = [0]*num_epochs
        self.stddev = [0]*num_epochs
        self._freeze()

    def record_stats(self, vals, epoch):
        """Record the stats of a set of activation values vals for an epoch."""
        self.mean[epoch] = np.mean(vals)
        self.stddev[epoch] = np.std(vals)



class TrainingHistory(FrozenObject):
    """
    Structure that contains a neural net's training history.
    """
    def __init__(self, nn, num_epochs):

        assert(isinstance(num_epochs, int) and num_epochs>0), \
            'Invalid num_epochs: {}'.format(num_epochs)
        self.num_epochs = num_epochs
        self.loss_history = [0]*num_epochs

        num_layers = nn.get_num_activation_layers()
        self.activation_history = [ActivationValuesHistory(num_epochs) for it
                                   in range(num_layers)]


    def record(self, loss, nn, epoch):
        """Record this epoch's training history."""
        self._check_epoch_input(epoch)
        self._record_epoch_loss(loss, epoch)
        self._record_epoch_activation(nn, epoch)


    def _record_epoch_loss(self, loss, epoch):
        """Record the loss after epoch."""
        self.loss_history[epoch] = loss


    def _record_epoch_activation(self, nn, epoch):
        """Record statistics about a set of activation_values."""
        li = 0 # activation layer index
        for l in nn.layers:
            if l.HAS_ACTIVATION:
                self.activation_history[li].record_stats(l.y, epoch)
                li = li + 1
        # Check that we did cover all activation layers.
        assert li == nn.get_num_activation_layers()


    def _check_epoch_input(self, epoch):
        """Check whether the user input epoch has a valid value."""
        assert(isinstance(epoch, int) and epoch >= 0
               and epoch < self.num_epochs), \
               'Invalid epoch input {}'.format(epoch)


    def plot_activation_history(self):
        """Plot self.mean and self.stddec visualization."""
        x = [it for it in range(self.num_epochs)]
        for li in range(len(self.activation_history)):
            y = self.activation_history[li].mean
            e = self.activation_history[li].stddev
            plt.errorbar(x, y, e)
            plt.ion()
            plt.pause(0.001)
        plt.show(block=True)
