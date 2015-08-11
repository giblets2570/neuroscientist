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
from recurrent_data import formatData
import json
import re
import math
import matplotlib.pyplot as plt
import os.path


import warnings
warnings.filterwarnings('ignore', '.*topo.*')
# warnings.simplefilter("error")

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

NUM_EPOCHS = 500

BATCH_SIZE = 26

NUM_HIDDEN_UNITS = 100
NUM_RECURRENT_UNITS = 200
LEARNING_RATE = 0.02
MOMENTUM = 0.9
GRAD_CLIP = 100

LOG_EXPERIMENT = True

TETRODE_NUMBER = 9

SAVE_MODEL = True

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    # the data is arranged as (num_sequences_per_batch, sequence_length, num_features_per_timestep)
    # num sequences per batch = batch size
    # sequence length = number of time steps per example. 
    # num_features_per_timestep = 31 i.e labels per tetrode

    X_train, X_valid, X_test, y_train, y_valid, y_test = formatData(sequenceLength=2000)

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

def model(input_shape, output_dim, num_hidden_units=NUM_HIDDEN_UNITS, num_recurrent_units=NUM_RECURRENT_UNITS, batch_size=BATCH_SIZE):
        """
            Create a symbolic representation of a neural network with `intput_dim`
            input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
            layer.
            The training function of this model must have a mini-batch size of
            `batch_size`.
            A theano expression which represents such a network is returned.

            Need to create a dense layer that converts the input to a 

        """
        length = input_shape[1]
        reduced_length = num_hidden_units

        shape = tuple([batch_size]+list(input_shape[1:]))
        print("Shape ",shape)

        # Construct vanilla RNN
        l_in = lasagne.layers.InputLayer(shape=shape)

        # print("Input shape: ",lasagne.layers.get_output_shape(l_in))

        l_reshape_1 = lasagne.layers.ReshapeLayer(l_in, (batch_size*length, input_shape[-1]))

        # print("Reshape 1 shape: ",lasagne.layers.get_output_shape(l_reshape_1))

        l_hidden_1 = lasagne.layers.DenseLayer(
            l_reshape_1,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        l_dropout_1 = lasagne.layers.DropoutLayer(
            l_hidden_1,
            p=0.8,
            )


        l_hidden_2 = lasagne.layers.DenseLayer(
            l_dropout_1,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        l_dropout_2 = lasagne.layers.DropoutLayer(
            l_hidden_2,
            p=0.8,
            )

        # print("Hidden 1 shape: ",lasagne.layers.get_output_shape(l_hidden_1))

        l_reshape_2 = lasagne.layers.ReshapeLayer(l_dropout_2, (batch_size, length, num_hidden_units))

        l_recurrent = lasagne.layers.GRULayer(
            l_reshape_2, num_hidden_units, 
            grad_clipping=GRAD_CLIP,
            gradient_steps=500,
            # W_in_to_hid=lasagne.init.HeUniform(),
            # W_hid_to_hid=lasagne.init.HeUniform(),
            # nonlinearity=lasagne.nonlinearities.sigmoid
            )

        print("Recurrent shape: ",lasagne.layers.get_output_shape(l_recurrent))

        # l_recurrent_back = lasagne.layers.GRULayer(
        #     l_in, num_hidden_units, 
        #     grad_clipping=GRAD_CLIP,
        #     gradient_steps=500,
        #     # W_in_to_hid=lasagne.init.HeUniform(),
        #     # W_hid_to_hid=lasagne.init.HeUniform(),
        #     # nonlinearity=lasagne.nonlinearities.sigmoid
        #     backwards=True
        #     )

        # print("Recurrent back shape: ",lasagne.layers.get_output_shape(l_recurrent_back))

        # l_recurrent_3 = lasagne.layers.GRULayer(
        #     l_recurrent_2, num_hidden_units, 
        #     grad_clipping=GRAD_CLIP,
        #     # W_in_to_hid=lasagne.init.HeUniform(),
        #     # W_hid_to_hid=lasagne.init.HeUniform(),
        #     # nonlinearity=lasagne.nonlinearities.sigmoid
        #     )


        # this is great

        # print("Recurrent 3 shape: ",lasagne.layers.get_output_shape(l_recurrent_3))

        # l_recurrent_back = lasagne.layers.RecurrentLayer(
        #     l_in, num_hidden_units, 
        #     grad_clipping=GRAD_CLIP,
        #     W_in_to_hid=lasagne.init.HeUniform(),
        #     W_hid_to_hid=lasagne.init.HeUniform(),
        #     nonlinearity=lasagne.nonlinearities.tanh, 
        #     backwards=True
        #     )

        # l_recurrent_2 = lasagne.layers.RecurrentLayer(
        #     l_recurrent, num_hidden_units, 
        #     grad_clipping=GRAD_CLIP,
        #     W_in_to_hid=lasagne.init.HeUniform(),
        #     W_hid_to_hid=lasagne.init.HeUniform(),
        #     nonlinearity=lasagne.nonlinearities.tanh
        #     )

        # l_recurrent_back_2 = lasagne.layers.RecurrentLayer(
        #     l_recurrent_back, num_hidden_units, 
        #     grad_clipping=GRAD_CLIP,
        #     W_in_to_hid=lasagne.init.HeUniform(),
        #     W_hid_to_hid=lasagne.init.HeUniform(),
        #     nonlinearity=lasagne.nonlinearities.tanh, 
        #     backwards=True
        #     )

        # l_recurrent_3 = lasagne.layers.RecurrentLayer(
        #     l_recurrent_2, num_hidden_units, 
        #     grad_clipping=GRAD_CLIP,
        #     W_in_to_hid=lasagne.init.HeUniform(),
        #     W_hid_to_hid=lasagne.init.HeUniform(),
        #     nonlinearity=lasagne.nonlinearities.tanh
        #     )
        

        # l_sum = lasagne.layers.ElemwiseSumLayer([l_recurrent, l_recurrent_back])

        # We need a reshape layer which combines the first (batch size) and second
        # (number of timesteps) dimensions, otherwise the DenseLayer will treat the
        # number of time steps as a feature dimension.
        l_reshape_3 = lasagne.layers.ReshapeLayer(l_recurrent, (batch_size*length, num_hidden_units))

        print("Reshape shape: ",lasagne.layers.get_output_shape(l_reshape_3))

        l_recurrent_out = lasagne.layers.DenseLayer(
            l_reshape_3,
            num_units=output_dim,
            nonlinearity=None
            )

        print("Recurrent out shape: ",lasagne.layers.get_output_shape(l_recurrent_out))

        l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
                                            (batch_size, length, output_dim))

        print("Output shape: ",lasagne.layers.get_output_shape(l_out))

        return l_out


def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM):

    """
        Method the returns the theano functions that are used in 
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables 
    X_batch = T.tensor3()
    y_batch = T.tensor3()
    l_rate = T.scalar()

    # this is the cost of the network when fed throught the noisey network
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.mse(train_output, y_batch)
    cost = cost.mean()

    # validation cost
    valid_output = lasagne.layers.get_output(network, X_batch, deterministic=True)
    valid_cost = lasagne.objectives.mse(valid_output, y_batch)
    valid_cost = valid_cost.mean()

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(network, X_batch, deterministic=True)
    pred = T.argmax(test, axis=1)
    accuracy = T.mean(lasagne.objectives.mse(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.adagrad(cost, all_params, l_rate)
    
    train = theano.function(inputs=[X_batch, y_batch, l_rate], outputs=cost, updates=updates, allow_input_downcast=True)
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

    print("Tetrode number: {}, Num outputs: {}".format(tetrode_number,dataset['output_dim']))

    print("Input shape: {}".format(dataset['X_train'].shape))
    print("Output shape: {}".format(dataset['y_train'].shape))
    
    print("Making the model...")
    network = model(dataset['input_shape'],dataset['output_dim'])
    print("Done!")

    if(os.path.isfile('recurrent_2_network')):
        print("Loading old model")
        f=open('recurrent_2_network','r')
        all_param_values = pickle.load(f)
        f.close()
        lasagne.layers.set_all_param_values(network, all_param_values)

    print("Setting up the training functions...")
    training = funcs(dataset,network)
    print("Done!")

    accuracies = []
    trainvalidation = []
    learning_rate = LEARNING_RATE

    print("Begining to train the network...")
    epochsDone = 0
    increasing = 0
    try:
        meanTrainCost = 1000
        for i in range(NUM_EPOCHS):
            costs = []
            valid_costs = []

            for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
                cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end],learning_rate)
                costs.append(cost)
                # if(costs[-1] > 1.02*costs[-2]):
                #     LEARNING_RATE = 0.8*LEARNING_RATE
                # print(cost)
            
            for start, end in zip(range(0, dataset['num_examples_valid'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_valid'], BATCH_SIZE)):
                cost = training['valid'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
                valid_costs.append(cost)

            if(np.mean(np.asarray(costs,dtype=np.float32)) > 1.00000001*meanTrainCost):
                increasing += 1
            else: 
                increasing = 0

            if increasing == 3:
                print("Lowering learning rate")
                learning_rate = 0.9*learning_rate
                increasing = 0
            meanValidCost = np.mean(np.asarray(valid_costs),dtype=np.float32) 
            meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
            # accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == np.argmax(training['predict'](dataset['X_test']), axis=1))

            print("Epoch: {}, Training cost: {}, Validation Cost: {}, learning rate: {}".format(i+1,meanTrainCost,meanValidCost,learning_rate))

            if(np.isnan(meanValidCost)):
                print("Nan value")
                break

            trainvalidation.append([meanTrainCost,meanValidCost])
            # accuracies.append(accuracy)

            epochsDone = epochsDone + 1

    except KeyboardInterrupt:
        pass

    # plt.plot(trainvalidation)
    # plt.show()

    if(LOG_EXPERIMENT):
        print("Logging the experiment details...")
        log = dict(
            NET_TYPE = "Recurent network 1 layer, {} units ".format(NUM_HIDDEN_UNITS),
            TETRODE_NUMBER = tetrode_number,
            BASENAME = BASENAME,
            NUM_EPOCHS = epochsDone,
            BATCH_SIZE = BATCH_SIZE,
            TRAIN_VALIDATION = trainvalidation,
            LEARNING_RATE = LEARNING_RATE,
            MOMENTUM = MOMENTUM,
            NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS,
            NETWORK_LAYERS = [str(type(layer)) for layer in lasagne.layers.get_all_layers(network)],
            OUTPUT_DIM = dataset['output_dim'],
            # NETWORK_PARAMS = lasagne.layers.get_all_params_values(network)
        )
        now = datetime.datetime.now()
        filename = "experiments/rec/{}_{}_{}_NUMLAYERS_{}_OUTPUTDIM_{}".format(now,NUM_EPOCHS,NUM_HIDDEN_UNITS,len(log['NETWORK_LAYERS']),log['OUTPUT_DIM'])
        filename = re.sub("[^A-Za-z0-9_/ ,-:]", "", filename)
        with open(filename,"w") as outfile:
            outfile.write(str(log))

    if(SAVE_MODEL):
        print("Saving model...")
        all_param_values = lasagne.layers.get_all_param_values(network)
        f=open('recurrent_2_network','w')
        pickle.dump(all_param_values, f)
        f.close()



if __name__ == '__main__':
  
    main()
