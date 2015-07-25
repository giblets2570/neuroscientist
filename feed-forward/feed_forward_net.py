#!/usr/bin/env python

"""Example which """

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
import datetime
from neuroscientist.data import formatData
import json
import re

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)


BASENAME = "../R2192/20140110_R2192_track1"

NUM_EPOCHS = 10
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = True
STOPPING_RANGE = 10

LOG_EXPERIMENT = False

TETRODE_NUMBER = 16

CONV = False

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    X_train, X_valid, X_test, y_train, y_valid, y_test = formatData(tetrode_number,BASENAME,CONV)

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    # y_train = y_train.reshape(y_train.shape[0],1,y_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
    # y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[1])

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_train.shape[0],
        num_examples_test=X_test.shape[0],
        input_shape=X_train.shape,
        output_dim=y_train.shape[-1],
    )

def model(input_shape, output_dim, num_hidden_units, p_drop_input, p_drop_hidden,batch_size=BATCH_SIZE):
	"""Create a symbolic representation of a neural network with `intput_dim`
    input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
    layer.
    The training function of this model must have a mini-batch size of
    `batch_size`.
    A theano expression which represents such a network is returned.
    """

        l_in = lasagne.layers.InputLayer(shape=input_shape)
        
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
        #     l_hidden_2,
        #     p=p_drop_hidden
        #     )
        l_hidden_3 = lasagne.layers.DenseLayer(
            l_hidden_2,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        # l_hidden_3_dropout = lasagne.layers.DropoutLayer(
        #     l_hidden_3,
        #     p=p_drop_hidden
        #     )
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
            l_hidden_3,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            )
        return l_out

def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM):

    """
        Method the returns the theano functions that are used in 
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables 
    X_batch = T.tensor3()
    y_batch = T.matrix()

    # this is the cost of the network when fed throught the noisey network
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.categorical_crossentropy(train_output, y_batch)
    cost = cost.mean()

    # validation cost
    valid_output = lasagne.layers.get_output(network, X_batch, deterministic=True)
    valid_cost = lasagne.objectives.categorical_crossentropy(valid_output, y_batch)
    valid_cost = valid_cost.mean()

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(network, X_batch, deterministic=True)
    pred = T.argmax(test, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate, momentum)
    
    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X_batch, y_batch], outputs=valid_cost, allow_input_downcast=True)
    predict = theano.function(inputs=[X_batch], outputs=test, allow_input_downcast=True)

    return dict(
        train=train,
        valid=valid,
        predict=predict
    )

def main(tetrode_number=TETRODE_NUMBER):
    """
        This is the main method that sets up the experiment
    """
    print("Loading the data...")
    dataset = load_data(tetrode_number)
    print("Done!")

    print(dataset['input_shape'])

    print("Making the model...")
    network = model(dataset['input_shape'],dataset['output_dim'],NUM_HIDDEN_UNITS,0.2,0.2)
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,network)
    print("Done!")

    accuracies = []


    print("Begining to train the network...")
    for i in range(NUM_EPOCHS):
        costs = []
        valid_costs = []

        for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
            cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end])
            costs.append(cost)
        
        for start, end in zip(range(0, dataset['num_examples_valid'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_valid'], BATCH_SIZE)):
            cost = training['valid'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
            valid_costs.append(cost)


        meanValidCost = np.mean(np.asarray(valid_costs),dtype=np.float32) 
        meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
        accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == training['predict'](dataset['X_test']))

        if(EARLY_STOPPING):
            if(len(accuracies) < STOPPING_RANGE):
                accuracies.append(accuracy)
            else:
                test = [k for k in accuracies if k < accuracy]
                if not test:
                    print('Early stopping causing training to finish at epoch {}'.format(i))
                    break
                del accuracies[0]
                accuracies.append(accuracy)

    if(LOG_EXPERIMENT):
        print("Logging the experiment details...")
        log = dict(
            TETRODE_NUMBER = tetrode_number,
            BASENAME = BASENAME,
            NUM_EPOCHS = NUM_EPOCHS,
            BATCH_SIZE = BATCH_SIZE,
            NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS,
            LEARNING_RATE = LEARNING_RATE,
            MOMENTUM = MOMENTUM,
            ACCURACY = accuracies[-1],
            NETWORK_LAYERS = [str(type(layer)) for layer in lasagne.layers.get_all_layers(network)],
            OUTPUT_DIM = dataset['output_dim'],
            # NETWORK_PARAMS = lasagne.layers.get_all_params_values(network)
        )
        now = datetime.datetime.now()
        filename = "experiments/{}_{}_{}_NUMLAYERS_{}_OUTPUTDIM_{}".format(now,NUM_EPOCHS,NUM_HIDDEN_UNITS,len(log['NETWORK_LAYERS']),log['OUTPUT_DIM'])
        filename = re.sub("[^A-Za-z0-9_/ ,-:]", "", filename)
        with open(filename,"w") as outfile:
            json.dump(log, outfile)



if __name__ == '__main__':
    # for i in range(16):
    #     main(i+1)
    main()
