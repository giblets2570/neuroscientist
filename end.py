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
from data import formatData as auto_data
from end_data import formatData as rec_data
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

NUM_EPOCHS = 2000

BATCH_SIZE = 26

NUM_HIDDEN_UNITS = 100
NUM_RECURRENT_UNITS = 200
LEARNING_RATE = 0.02
MOMENTUM = 0.9
GRAD_CLIP = 100

LOG_EXPERIMENT = True

TETRODE_NUMBER = 11

SAVE_MODEL = True

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    # the data is arranged as (num_sequences_per_batch, sequence_length, num_features_per_timestep)
    # num sequences per batch = batch size
    # sequence length = number of time steps per example.
    # num_features_per_timestep = 31 i.e labels per tetrode
    sequenceLength=500

    _,_,_, y_train, y_valid, y_test = rec_data(sequenceLength=sequenceLength)

    _,_time,_ = auto_data(tetrode_number,BASENAME,timed=True)

    time = []
    i = 0
    num_skip = 100
    print(_time.shape)
    while(i + sequenceLength < _time.shape[0]):
        time.append(_time[i:i+sequenceLength])
        # i+=25
        i+=num_skip
    time = np.asarray(time)
    print(time.shape)

    n = int(len(time)*0.8)
    m = int(len(time)*0.9)


    X_train = time[:n]
    X_valid = time[n:m]
    X_test = time[m:]

    # X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    # X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1])
    # X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

    print("X_TRAIN: {}".format(X_train.shape))

    print("y_TRAIN: {}".format(y_train.shape))

    return dict(
        X_train=time[:n],
        y_train=y_train,
        X_valid=time[n:m],
        y_valid=y_valid,
        X_test=time[m:],
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
        print(shape)


        l_in = lasagne.layers.InputLayer(shape=shape)

        print("In 1 shape: ",lasagne.layers.get_output_shape(l_in))

        l_reshape= lasagne.layers.ReshapeLayer(
            l_in,
            (batch_size*length,200)
            )

        l_hidden_1_1 = lasagne.layers.DenseLayer(
            l_reshape,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 1 1 shape: ",lasagne.layers.get_output_shape(l_hidden_1_1))

        l_code_layer_1 = lasagne.layers.DenseLayer(
            l_hidden_1_1,
            num_units=50,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            )

        print("code 1 1 shape: ",lasagne.layers.get_output_shape(l_code_layer_1))

        l_hidden_6_1 = lasagne.layers.DenseLayer(
            l_code_layer_1,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 6 1 shape: ",lasagne.layers.get_output_shape(l_hidden_6_1))

        l_almost = lasagne.layers.DenseLayer(
            l_hidden_6_1,
            num_units=200,
            nonlinearity=None,
            )

        l_out_1 = lasagne.layers.ReshapeLayer(
            l_almost,
            (batch_size,length,200)
            )

        print("Out 1 1 shape: ",lasagne.layers.get_output_shape(l_out_1))


        l_hidden_1 = lasagne.layers.DenseLayer(
            l_code_layer_1,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        print("Hidden 1 shape: ",lasagne.layers.get_output_shape(l_hidden_1))

        l_dropout = lasagne.layers.DropoutLayer(
            l_hidden_1,
            p=0.8,
            )


        l_hidden_2 = lasagne.layers.DenseLayer(
            l_dropout,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        print("Hidden 2 shape: ",lasagne.layers.get_output_shape(l_hidden_2))

        l_reshape_2 = lasagne.layers.ReshapeLayer(l_hidden_2, (batch_size, length, num_hidden_units))

        print("Reshape_2 shape: ",lasagne.layers.get_output_shape(l_reshape_2))

        l_recurrent = lasagne.layers.GRULayer(
            l_reshape_2, num_hidden_units, 
            grad_clipping=GRAD_CLIP,
            gradient_steps=500,
            # W_in_to_hid=lasagne.init.HeUniform(),
            # W_hid_to_hid=lasagne.init.HeUniform(),
            # nonlinearity=lasagne.nonlinearities.sigmoid
            )

        print("Recurrent shape: ",lasagne.layers.get_output_shape(l_recurrent))



        l_reshape_3 = lasagne.layers.ReshapeLayer(l_recurrent, (batch_size*length, num_hidden_units))

        print("Reshape 3 shape: ",lasagne.layers.get_output_shape(l_reshape_3))

        l_recurrent_out = lasagne.layers.DenseLayer(
            l_reshape_3,
            num_units=output_dim,
            nonlinearity=None
            )

        print("Recurrent out shape: ",lasagne.layers.get_output_shape(l_recurrent_out))

        l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
                                            (batch_size, length, output_dim))

        print("Output shape: ",lasagne.layers.get_output_shape(l_out))

        return l_out, l_out_1


def funcs(dataset, rec_network, auto_network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM, sparsity=0.01,beta=0.0002):

    """
        Method the returns the theano functions that are used in 
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables
    X_batch = T.tensor3()
    y_batch = T.tensor3()
    l_rate = T.scalar()

    rec_layers = lasagne.layers.get_all_layers(rec_network)
    num_rec_layers = len(rec_layers)
    # print(rec_layers)

    auto_layers = lasagne.layers.get_all_layers(auto_network)
    num_auto_layers = len(auto_layers)

    code_layer = auto_layers[num_auto_layers/2]
    # code outputs
    code_output = lasagne.layers.get_output(code_layer, X_batch, deterministic=True)

    # print(auto_layers)
    # this is the cost of the network when fed throught the noisey network
    auto_train_output = lasagne.layers.get_output(auto_network, X_batch)
    rec_train_output = lasagne.layers.get_output(rec_network, X_batch)
    auto_cost = lasagne.objectives.mse(auto_train_output, X_batch)
    rec_cost = lasagne.objectives.mse(rec_train_output, y_batch)

    rho_hat = T.mean(code_output,axis=1)
    L = T.sum(sparsity * T.log(sparsity/rho_hat) + (1 - sparsity) * T.log((1 - sparsity)/(1 - rho_hat)))


    cost = auto_cost.mean() + rec_cost.mean() + beta * sparsity

    # validation cost
    valid_output = lasagne.layers.get_output(rec_network, X_batch, deterministic=True)
    valid_cost = lasagne.objectives.mse(valid_output, y_batch)
    valid_cost = valid_cost.mean()

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(rec_network, X_batch, deterministic=True)

    all_params = lasagne.layers.get_all_params(rec_network) + lasagne.layers.get_all_params(auto_network)[2:]
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
    rec_network, auto_network = model(dataset['input_shape'],dataset['output_dim'])
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,rec_network, auto_network)
    print("Done!")

    if(os.path.isfile('end_network_auto_{}'.format(tetrode_number))):
        print("Loading old model")
        f=open('end_network_auto_{}'.format(tetrode_number),'r')
        all_param_values = pickle.load(f)
        f.close()
        lasagne.layers.set_all_param_values(auto_network, all_param_values)

    if(os.path.isfile('end_network_recurrent_{}'.format(tetrode_number))):
        print("Loading old model")
        f=open('end_network_recurrent_{}'.format(tetrode_number),'r')
        all_param_values = pickle.load(f)
        f.close()
        lasagne.layers.set_all_param_values(rec_network, all_param_values)

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

    predictions = []
    cost_arrays = []
    actuals = []
    for start, end in zip(range(0, dataset['num_examples_test'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_test'], BATCH_SIZE)):
        prediction = training['predict'](dataset['X_train'][start:end])
        predictions.append(prediction)
        # accuracy = np.mean(np.argmax(dataset['y_train'], axis=1) == np.argmax(training['predict'](dataset['X_train']), axis=1))
        actuals.append(dataset['y_train'][start:end])
    points_from = 300
    for i,(actual,prediction) in enumerate(zip(actuals,predictions)):
        prediction = np.asarray(prediction)
        actual = np.asarray(actual)

        print("Actual: {}".format(actual.shape))
        print("Prediction: {}".format(prediction.shape))
        dist = np.linalg.norm(actual-prediction)
        print("Distance: {}".format(dist))

        fig = plt.figure(1)

        sub1 = fig.add_subplot(121)
        sub2 = fig.add_subplot(122)

        sub1.set_title("Predicted", fontsize=16)
        sub2.set_title("Actual", fontsize=16)
        sub1.scatter(prediction[0,points_from:,0],prediction[0,points_from:,1],lw=0.0)
        sub1.axis([0.0,1.0,0.0,1.0])
        sub2.scatter(actual[0,points_from:,0],actual[0,points_from:,1],c=(1,0,0,1),lw=0.2)
        sub2.axis([0.0,1.0,0.0,1.0])
        sub1.grid(True)
        sub2.grid(True)

        fig.tight_layout()

        plt.savefig('../position/test/End_Position_{}.png'.format(i), bbox_inches='tight')
        plt.close()

    if(SAVE_MODEL):
        print("Saving model...")
        all_param_values = lasagne.layers.get_all_param_values(auto_network)
        f=open('end_network_auto_{}'.format(tetrode_number),'w')
        pickle.dump(all_param_values, f)
        f.close()
        all_param_values = lasagne.layers.get_all_param_values(rec_network)
        f=open('end_network_recurrent_{}'.format(tetrode_number),'w')
        pickle.dump(all_param_values, f)
        f.close()

if __name__ == '__main__':
    main()