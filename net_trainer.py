import numpy as np
import mlp, sequential_net


class NetTrainer:
    """
    Train a neural net.
    """

    def sgd(self, nn, x_train, y_train, epochs, batch_size, eta=None):
        """Train a neural net nn using batched stochastic gradient descent, return
        the trained neural net."""
        # eta is the learning rate.

        # Check input argument consistency
        assert(isinstance(nn, mlp.MLP) or \
               isinstance(nn, sequential_net.SequentialNet)), \
            'Input neural net nn is not an instance of MLP class or '\
            'SequentialNet class. nn.__class__ = {}'.format(nn.__class__)

        assert(x_train.shape[0] == y_train.shape[0]), \
            'x_train and y_train should have the same number of samples.'

        input_dims = nn.get_layer_dims(0)
        assert(x_train.shape[1:] == input_dims), \
            'x_train data has dimension {}, inconsistent with the neural net\'s '\
            'input dimension {}.'.format(
                x_train.shape[1:], input_dims)

        output_dims = nn.get_layer_dims(-1)
        assert(y_train.shape[1:] == output_dims), \
            'y_train data has dimension {}, inconsistent with the neural net\'s '\
            'output dimension {}.'.format(
                y_train.shape[1:], output_dims)

        num_samples = x_train.shape[0]
        assert(batch_size <= num_samples), 'batch_size [{}] > number of samples '\
            'in x/y_train [{}].'.format(
                batch_size, num_samples)

        # Initialize loss_history
        loss_history = []

        # Initialize learning rate
        eta = self._get_etas(epochs, eta)

        # Do batched sgd
        for i in range(epochs):
            nn.reset_cache()
            cumulative_loss = 0
            cumulative_loss_gradient = np.zeros(output_dims)
            loss_grads = np.zeros((batch_size, *output_dims))

            # Evaluate loss and loss gradient for a batch
            for j in range(batch_size):
                s = self._select_sample(j, num_samples)
                res = nn.forward_pass(x_train[s])
                cumulative_loss += nn.evaluate_loss(res, y_train[s])
                loss_grad = nn.calculate_loss_gradient(res, y_train[s])
                nn.backprop(loss_grad)

            # Train for this epoch
            cumulative_loss = cumulative_loss / batch_size
            nn.update_weights(batch_size, eta[i])
            #weights_before = nn.get_layer(1).get_weights()
            #weights_after = nn.get_layer(1).get_weights()
            #delta_w = weights_after - weights_before
            #plt.imshow(delta_w)
            #plt.show()

            #print('Epoch #%d: loss = %f\n' % (i, cumulative_loss))
            loss_history.append(cumulative_loss)

        return nn, loss_history


    def _select_sample(self, count, num_samples):
        """Helper function to select sample from num_samples."""
        return np.random.randint(low=0, high=num_samples-1)


    def _get_etas(self, epochs, eta):
        assert(epochs>0), 'num epochs must be >0.'
        if eta is None:
            eta_init = 0.01
            eta_prelim = [eta_init**(2**-n) for n in range(epochs)]
            return [it if it > 1 else 1 for it in eta_prelim]
        else:
            return [eta]*epochs


    def evaluate(self, nn, x_test, y_test):
        """Evaluate the neural network nn against the test data set."""
        assert(x_test.shape[0] == y_test.shape[0]), \
            'x_test.shape = {}, y_test.shape = {}, inconsistent sizes.'\
            .format(x_test.shape, y_test.shape)
        num_tests = x_test.shape[0]
        num_correct = 0
        cumulative_loss = 0

        # Check test samples
        for i in range(num_tests):
            res, loss = nn.classify_one_hot(x_test[i], y_test[i])
            cumulative_loss += loss
            if self._compare_results(res, y_test[i]):
                num_correct += 1

        # Compile total stats
        cumulative_loss = cumulative_loss / num_tests
        accuracy = num_correct * 1.0 / num_tests
        return accuracy, cumulative_loss


    def _compare_results(self, y0, y1):
        """Helper function to compare 2 results, ignoring any formatting or data
        type differences"""
        # Currently only implemented for one-hot encoded classification output.
        # Augment as needed.
        y0 = y0.astype(int)
        y1 = y1.astype(int)
        return np.sum(y0)==np.sum(y1) and np.sum(y0) == 1 and \
            np.where(y0==1) == np.where(y1==1)
