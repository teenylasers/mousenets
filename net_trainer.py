import numpy as np
import mlp, sequential_net
from training_history import *


class NetTrainer:
    """
    Train a neural net.
    """

    def sgd(self, nn, x_train, y_train, epochs, batch_size, num_batches=None,
            eta=None, viz=False):
        """Train a neural net nn using batched stochastic gradient descent, return
        the trained neural net. If viz, plot gradient distribution at each layer.
        """
        # Check input arguments are valid
        self._check_input(nn, x_train, y_train, batch_size)

        # Initialize learning rate eta
        eta = self._get_etas(epochs, eta)

        # Initialize training_history object
        history = TrainingHistory(nn, epochs)

        # Do batched sgd
        output_dims = nn.get_layer_dims(-1)
        num_samples = x_train.shape[0]

        for i in range(epochs):

            batches = self._form_batches(num_samples, batch_size, num_batches)
            cumulative_loss = 0

            print('Epoch {}: '.format(i), end='')
            for j in range(len(batches)):

                nn.reset_cache()

                # Evaluate loss and loss gradient for a batch
                for s in batches[j]:
                    res = nn.forward_pass(x_train[s])
                    cumulative_loss += nn.evaluate_loss(res, y_train[s])
                    loss_grad = nn.calculate_loss_gradient(res, y_train[s])
                    nn.backprop(loss_grad)

                # Train for this epoch
                #if viz:
                #    nn.plot_gradient_distribution()
                nn.update_weights(batch_size, eta[i])
                print('=',end='')
                #weights_before = nn.get_layer(1).get_weights()
                #weights_after = nn.get_layer(1).get_weights()
                #delta_w = weights_after - weights_before
                #plt.imshow(delta_w)
                #plt.show()

            # Record the ending cumulative loss and neural net state in this
            # epoch
            cumulative_loss = cumulative_loss / sum([len(it) for it in batches])
            print('| cumulative_loss = {}'.format(cumulative_loss))
            history.record(cumulative_loss, nn, i)

        # Visualize activation values history, if viz
        #if viz:
        #    history.plot_activation_history()

        return nn, history


    def _check_input(self, nn, x_train, y_train, batch_size):
        """Check that the input arguments to a NetTrainer are valid. """

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


    def _form_batches(self, num_samples, batch_size, num_batches=None):
        """Form a list of batches, which of length batch_size from a total
        num_samples. Each batch is a list of sample indices. If num_batches is
        not specified, split all samples into batches. Else, list length is
        num_batches."""
        n = int(np.ceil(num_samples / batch_size))
        if num_batches is None:
            num_batches = n
        else:
            assert(num_batches <= n), 'Too many num_batches [{}] for num_samples'\
                ' [{}] and batch_size [{}]'.format(
                    num_batches, num_samples, batch_size)
        samples = list(range(num_samples))
        np.random.shuffle(samples)
        if num_batches < n:
            res = [samples[it*batch_size : it*batch_size+batch_size]
                   for it in range(num_batches)]
        else:
            res = [samples[it*batch_size : it*batch_size+batch_size]
                   for it in range(num_batches-1)]
            res.append(samples[num_batches-1*batch_size :])
        return res


    def _get_etas(self, epochs, eta):
        """Return a learning rate schedule, a list of eta for each training epoch."""
        assert(epochs>0), 'num epochs must be >0.'
        if eta is None:
            eta_init = 0.9
            eta_prelim = [eta_init/(2**n) for n in range(epochs)]
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
