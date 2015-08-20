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
from big_data import formatData
import json
import re
from pos_data import getXY

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)


BASENAME = "../R2192-screening/20141001_R2192_screening"

NUM_EPOCHS = 1000000
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = False
STOPPING_RANGE = 10

LOG_EXPERIMENT = True

TETRODE_NUMBER = 9

L2_CONSTANT = 0.0001

CONV = False

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    # X_train, X_valid, X_test= formatData(BASENAME)

    # X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    # X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1])
    # # y_train = y_train.reshape(y_train.shape[0],1,y_train.shape[1])
    # X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
    # y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[1])

    x, y = getXY()

    m = int(len(x)*0.8)
    n = int(len(x)*0.9)
    
    y_train = np.asarray([x[:m],y[:m]])
    y_valid = np.asarray([x[m:n],y[m:n]])
    y_test = np.asarray([x[n:],y[n:]])

    y_train = y_train.transpose()
    y_valid = y_valid.transpose()
    y_test = y_test.transpose()

    print(y_train.shape)

    X_train, X_valid, X_test= formatData(BASENAME)

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_shape=X_train.shape,
        output_dim=y_train.shape[-1],
    )

def model(input_shape, output_dim,batch_size=BATCH_SIZE):
    """Create a symbolic representation of a neural network with `intput_dim`
    input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
    layer.
    The training function of this model must have a mini-batch size of
    `batch_size`.
    A theano expression which represents such a network is returned.
    """

    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE,input_shape[-1]))

    print("Input shape: ",lasagne.layers.get_output_shape(l_in))

    # l_in_dropout = lasagne.layers.DropoutLayer(
    #     l_in,
    #     p=p_drop_input
    #     )

    l_hidden = lasagne.layers.DenseLayer(
        l_in,
        num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    print("Hidden 1 shape: ",lasagne.layers.get_output_shape(l_hidden))

    # l_hidden_dropout = lasagne.layers.DropoutLayer(
    #     l_hidden,
    #     p=0.8
    #     )

    # # print("Hidden drop 1 shape: ",lasagne.layers.get_output_shape(l_hidden_dropout))

    l_hidden_2 = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=600,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    # print("Hidden 2 shape: ",lasagne.layers.get_output_shape(l_hidden_2))

    # l_hidden_2_dropout = lasagne.layers.DropoutLayer(
    #     l_hidden_2,
    #     p=0.8
    #     )

    l_hidden_3 = lasagne.layers.DenseLayer(
        l_hidden_2,
        num_units=200,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    # print("Hidden 3 shape: ",lasagne.layers.get_output_shape(l_hidden_3))

    # l_hidden_3_dropout = lasagne.layers.DropoutLayer(
    #     l_hidden_3,
    #     p=p_drop_hidden
    #     )

    l_hidden_4 = lasagne.layers.DenseLayer(
        l_hidden_3,
        num_units=50,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    l_out = lasagne.layers.DenseLayer(
        l_hidden_2,
        num_units=output_dim,
        nonlinearity=None
        )

    print("Output shape: ",lasagne.layers.get_output_shape(l_out))

    return l_out

def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM, alpha=L2_CONSTANT):

    """
        Method the returns the theano functions that are used in
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables
    X_batch = T.matrix()
    y_batch = T.matrix()

    # this is the cost of the network when fed throught the noisey network
    l2 = lasagne.regularization.l2(X_batch)
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.mse(train_output, y_batch)
    cost = cost.mean() #+ alpha*l2

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(network, X_batch, deterministic=True)
    pred = T.argmax(test, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate, momentum)

    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X_batch, y_batch], outputs=cost, allow_input_downcast=True)
    predict = theano.function(inputs=[X_batch], outputs=pred, allow_input_downcast=True)

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
    network = model(dataset['input_shape'],dataset['output_dim'])
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,network)
    print("Done!")

    accuracies = []


    print("Begining to train the network...")
    try:

        for i in range(NUM_EPOCHS):
            costs = []
            valid_costs = []

            for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
                cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end])
                if (np.isnan(cost)):
                    print(":)")
                else:
                    costs.append(cost)
            for start, end in zip(range(0, dataset['num_examples_valid'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_valid'], BATCH_SIZE)):
                cost = training['valid'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
                if (np.isnan(cost)):
                    print(":o")
                else:
                    valid_costs.append(cost)

            # accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == training['predict'](dataset['X_test']))

            meanValidCost = np.mean(np.asarray(valid_costs),dtype=np.float32)
            meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))

            # print("Epoch: {}, Accuracy: {}".format(i+1,accuracy))
            print("Epoch: {}, Training cost: {}, validation cost: {}".format(i+1,meanTrainCost,meanValidCost))

            if(np.isnan(meanValidCost)):
                print("Nan value")
                break

            # if(EARLY_STOPPING):
            #     if(len(accuracies) < STOPPING_RANGE):
            #         accuracies.append(accuracy)
            #     else:
            #         test = [k for k in accuracies if k < accuracy]
            #         if not test:
            #             print('Early stopping causing training to finish at epoch {}'.format(i))
            #             break
            #         del accuracies[0]
            # accuracies.append(accuracy)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    # for i in range(16):
    #     main(i+1)
    main()