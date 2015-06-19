#!/usr/bin/env python

"""Example which shows with the MNIST dataset how Lasagne can be used."""

from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from data import formatData

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)


BASENAME = "../../R2192/20140110_R2192_track1"

NUM_EPOCHS = 1000
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EARLY_STOPPING = True
STOPPING_RANGE = 10

def floatX(X):
    """
        Returns the correct data format
    """
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    """
        This function initialises the weight of the network
    """
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def load_data():
    """
        Get data with labels, split into training and test set.
    """
    trX, teX, trY, teY = formatData(16,BASENAME)
    X_train, y_train = trX, trY
    X_test, y_test = teX, teY

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
    )

def model(input_dim, output_dim, num_hidden_units, p_drop_input, p_drop_hidden,batch_size=BATCH_SIZE):
	"""Create a symbolic representation of a neural network with `intput_dim`
    input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
    layer.
    The training function of this model must have a mini-batch size of
    `batch_size`.
    A theano expression which represents such a network is returned.
    """

        l_in = lasagne.layers.InputLayer(shape=(batch_size, input_dim))
        
	# l_in_dropout = lasagne.layers.DropoutLayer(l_in,p=p_drop_input)

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        # l_hidden_dropout = lasagne.layers.DropoutLayer(
        #     l_hidden,
        #     p=p_drop_hidden
        #     )
        l_hidden_2 = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        # l_hidden_2_dropout = lasagne.layers.DropoutLayer(
        #     l_hidden,
        #     p=p_drop_hidden
        #     )
        # l_hidden_3 = lasagne.layers.DenseLayer(
        #     l_hidden_2,
        #     num_units=num_hidden_units,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     )
        # # l_hidden_3_dropout = lasagne.layers.DropoutLayer(
        # #     l_hidden_3,
        # #     p=p_drop_hidden
        # #     )
        # l_hidden_4 = lasagne.layers.DenseLayer(
        #     l_hidden_3,
        #     num_units=num_hidden_units,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     )
        # l_hidden_4_dropout = lasagne.layers.DropoutLayer(
        #     l_hidden_4,
        #     p=p_drop_hidden
        #     )
	l_out = lasagne.layers.DenseLayer(
            l_hidden_2,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            )
        return l_out

def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM):

    # symbolic variables 
    X_batch = T.matrix()
    y_batch = T.matrix()

    # this is the cost of the network when fed throught the noisey network
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.categorical_crossentropy(train_output, y_batch)
    cost = cost.mean()

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(network, X_batch, deterministic=True)
    pred = T.argmax(test, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate, momentum)
    
    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X_batch], outputs=pred, allow_input_downcast=True)

    return dict(
        train=train,
        predict=predict
    )

def main():
    dataset = load_data()

    print("Making networks with shared weights")
    network = model(dataset['input_dim'],dataset['output_dim'],NUM_HIDDEN_UNITS,0.2,0.2)
    # cleanNetwork = model(dataset['input_dim'],dataset['output_dim'],NUM_HIDDEN_UNITS,0.,0.,w_h1,w_h2,w_h3,w_h4,w_o)
    print("Done")

    training = funcs(dataset,network)

    accuracies = []

    for i in range(NUM_EPOCHS):
        for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
            cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end])
        accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == training['predict'](dataset['X_test']))
        print(accuracy)
        if(EARLY_STOPPING):
            if(len(accuracies) < STOPPING_RANGE):
                accuracies.append(accuracy)
            else:
                test = [k for k in accuracies if k < accuracy]
                if not test:
                    print('Early stopping causing training to finish at epoch {}'.format(k))
                    break
                del accuracies[0]
                accuracies.append(accuracy)


if __name__ == '__main__':
    main()
