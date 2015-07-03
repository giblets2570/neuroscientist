'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``
    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (n_batch, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (n_batch,).
    mask : np.ndarray
        A binary matrix of shape (n_batch, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.
    References
    ----------
    .. [1] Hochreiter, Sepp, and Jurgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    mask = np.zeros((n_batch, max_length))
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype(theano.config.floatX))


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
    # We'll use an elementwise sum to combine the forward/backward layers
    l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
    # We need a reshape layer which combines the first (batch size) and second
    # (number of timesteps) dimensions, otherwise the DenseLayer will treat the
    # number of time steps as a feature dimension
    l_reshape = lasagne.layers.ReshapeLayer(
        l_sum, (N_BATCH*MAX_LENGTH, N_HIDDEN))
    # Our output layer is a simple dense connection, with 1 output unit
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (N_BATCH, MAX_LENGTH))

    target_values = T.vector('target_output')
    mask = T.matrix('mask')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(
        l_out, l_in.input_var, mask=mask)
    # The value we care about is the final value produced for each sequence
    predicted_values = network_output[:, -1]
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function(
        [l_in.input_var, target_values, mask], cost, updates=updates)
    compute_cost = theano.function([l_in.input_var, target_values, mask], cost)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data()

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                X, y, m = gen_data()
                train(X, y, m)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()