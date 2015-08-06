#!/usr/bin/env python

"""Example which """

from __future__ import print_function

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

from random import randint
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
import math
import matplotlib.pyplot as plt
from tsne import bh_sne
from itertools import cycle

from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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
BATCH_SIZE = 400
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = False
STOPPING_RANGE = 10

LOG_EXPERIMENT = True

TETRODE_NUMBER = 9

CONV = False

SAVE_MODEL = True

class DimshuffleLayer(lasagne.layers.Layer):
    def __init__(self, input_layer, pattern):
        super(DimshuffleLayer, self).__init__(input_layer)
        self.pattern = pattern

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(self.pattern)

def load_data(tetrodeRange=[TETRODE_NUMBER]):
    """
        Get data with labels, split into training and test set.
    """
    print("Loading data...")
    X_train, X_valid, X_test, y_train_labels, y_valid_labels, y_test_labels = formatData(BASENAME,tetrodeRange)
    print("Done!")

    # X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    # X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1])
    # X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])


    y_train = X_train
    y_valid = X_valid
    y_test = X_test

    r={}
    for x,y in zip(X_test,y_test_labels):
        # print("x: {}".format(x))
        # print("y: {}".format(y))
        _y = list(y)
        if int(_y.index(1.0)) not in r:
            r[int(_y.index(1.0))]=[x]
        else:
            r[int(_y.index(1.0))].append(x)

    for key in r:
        r[key] = np.asarray(r[key])


    return dict(
        X_train=X_train,
        y_train=y_train,
        y_train_labels=[np.argmax(y) for  y in y_train_labels],
        X_valid=X_valid,
        y_valid=y_valid,
        y_valid_labels=[np.argmax(y) for  y in y_valid_labels],
        X_test=X_test,
        y_test=y_test,
        y_test_labels=[np.argmax(y) for  y in y_test_labels],
        labeled_test=r,
        caswells_dim = y_train_labels.shape[-1],
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_shape=X_train.shape,
        output_dim=y_train.shape[-1],
    )


def model(input_shape, output_dim, num_hidden_units,num_hidden_units_2, num_code_units, batch_size=BATCH_SIZE):
        """
            Create a symbolic representation of a neural network with `intput_dim`
            input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
            layer.
            The training function of this model must have a mini-batch size of
            `batch_size`.
            A theano expression which represents such a network is returned.
        """
        shape = tuple([None]+list(input_shape[1:]))
        print(shape)
        l_in = lasagne.layers.InputLayer(shape=shape)

        l_hidden_1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_hidden_2 = lasagne.layers.DenseLayer(
            l_hidden_1,
            num_units=num_hidden_units_2,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_code_layer = lasagne.layers.DenseLayer(
            l_hidden_2,
            num_units=num_code_units,
            nonlinearity=lasagne.nonlinearities.softmax,
            )

        l_hidden_3 = lasagne.layers.DenseLayer(
            l_code_layer,
            num_units=num_hidden_units_2,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_hidden_4 = lasagne.layers.DenseLayer(
            l_hidden_3,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_out = lasagne.layers.DenseLayer(
            l_hidden_4,
            num_units=output_dim,
            nonlinearity=None,
            )

        return l_out

def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, sparsity=0.02, beta=0.1, momentum=MOMENTUM):

    """
        Method the returns the theano functions that are used in 
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables 
    X_batch = T.matrix()
    y_batch = T.matrix()

    layers = lasagne.layers.get_all_layers(network)
    num_layers = len(layers)
    print(num_layers)

    code_layer = layers[num_layers/2]

    # code output 
    code_output = lasagne.layers.get_output(code_layer, X_batch, deterministic=True)

    l = T.sub(1,code_output)
    ll = T.mul(code_output,l)
    L = T.mul(4,ll)
    L = L.mean()

    rho_hat = T.mean(code_output,axis=1)
    # L = T.sum(sparsity * T.log(sparsity/rho_hat) + (1 - sparsity) * T.log((1 - sparsity)/(1 - rho_hat)))

    # reg = 0.0001*lasagne.regularization.l2(network)
    # this is the cost of the network when fed throught the noisey network
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.mse(train_output, y_batch) 
    cost = cost.mean() + beta * L
    # validation cost
    valid_output = lasagne.layers.get_output(network, X_batch)
    valid_cost = lasagne.objectives.mse(valid_output, y_batch) 
    valid_cost = valid_cost.mean() 

    # test the performance of the netowork without noise
    pred = lasagne.layers.get_output(network, X_batch, deterministic=True)
    # pred = T.argmax(test, axis=1)
    accuracy = 1 - T.mean(lasagne.objectives.mse(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate, momentum)

    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X_batch, y_batch], outputs=valid_cost, allow_input_downcast=True)
    predict = theano.function(inputs=[X_batch], outputs=pred, allow_input_downcast=True)
    accuracy = theano.function(inputs=[X_batch,y_batch], outputs=accuracy, allow_input_downcast=True)
    code = theano.function(inputs=[X_batch], outputs=code_output, allow_input_downcast=True)
    L_penalty = theano.function(inputs=[X_batch], outputs=L, allow_input_downcast=True)

    return dict(
        train=train,
        valid=valid,
        predict=predict,
        accuracy=accuracy,
        code=code,
        L_penalty=L_penalty
    )

def main(tetrode_number=TETRODE_NUMBER,num_hidden_units=600,num_hidden_units_2=400,num_code_units=200):
    """
        This is the main method that sets up the experiment
    """

    print("Loading the data...")
    dataset = load_data([tetrode_number])
    print("Done!")

    print("Tetrode number: {}, Num outputs: {}".format(tetrode_number,dataset['output_dim']))

    print(dataset['input_shape'])
    print(dataset['output_dim'])
    
    print("Making the model...")
    network = model(dataset['input_shape'],dataset['output_dim'],num_hidden_units,num_hidden_units_2,num_code_units)
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,network)
    print("Done!")

    accuracies = []
    trainvalidation = []

    print("Begining to train the network...")
    epochsDone = 0
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


            meanValidCost = np.mean(np.asarray(valid_costs),dtype=np.float32) 
            meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
            accuracy = training['accuracy'](dataset['X_test'],dataset['y_test'])

            print("Epoch: {}, Accuracy: {}, Training cost: {}, validation cost: {}".format(i+1,accuracy,meanTrainCost,meanValidCost))

            if(np.isnan(meanTrainCost/meanValidCost)):
                print("Nan value")
                break

            trainvalidation.append([meanTrainCost,meanValidCost])
            accuracies.append(accuracy)
            if(EARLY_STOPPING):
                if(len(accuracies) < STOPPING_RANGE):
                    pass
                else:
                    test = [k for k in accuracies if k < accuracy]
                    if not test:
                        print('Early stopping causing training to finish at epoch {}'.format(i+1))
                        break
                    del accuracies[0]
                    accuracies.append(accuracy)

            epochsDone = epochsDone + 1

            if(i%100 == 0):
                tetrode_number += 1
                if(tetrode_number > 16):
                    tetrode_number = 9
                print("Loading the data...")
                dataset = load_data([tetrode_number])
                print("Done!")

    except KeyboardInterrupt:
        pass

    # plt.plot(trainvalidation)
    # plt.show()

    if(LOG_EXPERIMENT):
        print("Logging the experiment details...")
        log = dict(
            NET_TYPE = "Auto encoder 2 hidden 1 code 300 200 100",
            TETRODE_NUMBER = tetrode_number,
            BASENAME = BASENAME,
            NUM_EPOCHS = epochsDone,
            BATCH_SIZE = BATCH_SIZE,
            TRAIN_VALIDATION = trainvalidation,
            LEARNING_RATE = LEARNING_RATE,
            MOMENTUM = MOMENTUM,
            ACCURACY = accuracies,
            NETWORK_LAYERS = [str(type(layer)) for layer in lasagne.layers.get_all_layers(network)],
            OUTPUT_DIM = dataset['output_dim'],
            # NETWORK_PARAMS = lasagne.layers.get_all_params_values(network)
        )
        now = datetime.datetime.now()
        filename = "experiments/big_auto/{}_{}_{}_NUMLAYERS_{}_OUTPUTDIM_{}".format(now,NUM_EPOCHS,NUM_HIDDEN_UNITS,len(log['NETWORK_LAYERS']),log['OUTPUT_DIM'])
        filename = re.sub("[^A-Za-z0-9_/ ,-:]", "", filename)
        with open(filename,"w") as outfile:
            outfile.write(str(log))

    if(SAVE_MODEL):
        print("Saving model...")
        all_param_values = lasagne.layers.get_all_param_values(network)
        f=open('big_auto_network','w')
        pickle.dump(all_param_values, f)
        f.close()

if __name__ == '__main__':
    main()
    
