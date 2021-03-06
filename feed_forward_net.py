#!/usr/bin/env python

"""Example which """

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
from data import formatData
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


BASENAME = "../R2192-screening/20141001_R2192_screening"

NUM_EPOCHS = 1000000
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = True
STOPPING_RANGE = 20

LOG_EXPERIMENT = True

TETRODE_NUMBER = 9

L2_CONSTANT = 0.001

# Epoch: 246, Accuracy: 0.943329015544, Training cost: 0.109099879861, validation cost: 0.17596784234
# 20

CONV = False

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    X_train, X_valid, X_test, y_train, y_valid, y_test = formatData(tetrode_number,BASENAME,CONV)

    # X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    # X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1])
    # # y_train = y_train.reshape(y_train.shape[0],1,y_train.shape[1])
    # X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
    # y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[1])

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

def model(input_shape, output_dim, num_hidden_units, p_drop_input, p_drop_hidden,batch_size=BATCH_SIZE):
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
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    print("Hidden 1 shape: ",lasagne.layers.get_output_shape(l_hidden))

    # l_hidden_dropout = lasagne.layers.DropoutLayer(
    #     l_hidden,
    #     p=p_drop_hidden
    #     )

    # # # print("Hidden drop 1 shape: ",lasagne.layers.get_output_shape(l_hidden_dropout))

    l_hidden_2 = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    # print("Hidden 2 shape: ",lasagne.layers.get_output_shape(l_hidden_2))

    # l_hidden_2_dropout = lasagne.layers.DropoutLayer(
    #     l_hidden_2,
    #     p=p_drop_hidden
    #     )

    # l_hidden_3 = lasagne.layers.DenseLayer(
    #     l_hidden_2,
    #     num_units=num_hidden_units,
    #     nonlinearity=lasagne.nonlinearities.rectify,
    #     )

    # print("Hidden 3 shape: ",lasagne.layers.get_output_shape(l_hidden_3))

    # l_hidden_3_dropout = lasagne.layers.DropoutLayer(
    #     l_hidden_3,
    #     p=p_drop_hidden
    #     )

    l_out = lasagne.layers.DenseLayer(
        l_hidden_2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
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
    all_params = lasagne.layers.get_all_params(network)

    l2 = lasagne.regularization.regularize_network_params(network,lasagne.regularization.l2)
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.categorical_crossentropy(train_output, y_batch)
    cost = cost.mean() + alpha*l2

    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate, momentum)

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(network, X_batch, deterministic=True)
    pred = T.argmax(test, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X_batch, y_batch], outputs=cost, allow_input_downcast=True)
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
    network = model(dataset['input_shape'],dataset['output_dim'],NUM_HIDDEN_UNITS,0.8,0.5)
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,network)
    print("Done!")

    accuracies = []

    trainValid=[]

    k = 0
    print("Begining to train the network...")
    try:

        for i in range(NUM_EPOCHS):
            costs = []
            valid_costs = []

            for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
                cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end])
                costs.append(cost)
            for start, end in zip(range(0, dataset['num_examples_valid'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_valid'], BATCH_SIZE)):
                cost = training['valid'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
                valid_costs.append(cost)

            accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == np.argmax(training['predict'](dataset['X_test']), axis=1))

            meanValidCost = np.mean(np.asarray(valid_costs),dtype=np.float32)
            meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
            trainValid.append([meanTrainCost,meanValidCost])

            # print("Epoch: {}, Accuracy: {}".format(i+1,accuracy))
            print("Epoch: {}, Accuracy: {}, Training cost: {}, validation cost: {}".format(i+1,accuracy,meanTrainCost,meanValidCost))

            if(np.isnan(meanValidCost)):
                print("Nan value")
                break

            if(EARLY_STOPPING):
                if(len(trainValid) > 2):
                    if(trainValid[-2][1] < trainValid[-1][1]):
                        k += 1
                    else:
                        k = 0
                    print(k)
                    if (k == STOPPING_RANGE):
                        raise(KeyboardInterrupt)

            accuracies.append(accuracy)

    except KeyboardInterrupt:
        pass


    code = training['predict']([dataset['X_test'][1000]])

    x_axis = list(np.arange(len(code[0])))

    # sub3.plot(code[0])
    plt.bar(x_axis, code[0], width=1)
    plt.title("Feed forward net output")
    plt.savefig("feed_ex.png")
    plt.close()

    if(LOG_EXPERIMENT):
        print("Logging the experiment details...")
        log = dict(
            TETRODE_NUMBER = tetrode_number,
            BASENAME = BASENAME,
            NUM_EPOCHS = NUM_EPOCHS,
            BATCH_SIZE = BATCH_SIZE,
            TRAIN_VALID = str(trainValid),
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