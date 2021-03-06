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
from big_data import formatData as auto_data
from pos_data import getXY as rec_data
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

EARLY_STOPPING = True
STOPPING_RANGE = 10

TETRODE_NUMBER = 11

SAVE_MODEL = True

L2_CONSTANT = 0.00000

def load_data(tetrodeRange=[11,12]):
    """
        Get data with labels, split into training and test set.
    """

    # the data is arranged as (num_sequences_per_batch, sequence_length, num_features_per_timestep)
    # num sequences per batch = batch size
    # sequence length = number of time steps per example.
    # num_features_per_timestep = 31 i.e labels per tetrode
    sequenceLength=500

    y_train, y_valid, y_test = rec_data(sequenceLength=sequenceLength)

    # _,_time,_ = auto_data(tetrode_number,BASENAME,timed=True)

    # time = []
    # i = 0
    # num_skip = 40
    # print(_time.shape)
    # while(i + sequenceLength < _time.shape[0]):
    #     time.append(_time[i:i+sequenceLength])
    #     # i+=25
    #     i+=num_skip
    # time = np.asarray(time)
    # print(time.shape)

    # n = int(len(time)*0.8)
    # m = int(len(time)*0.9)


    # X_train = time[:n]
    # X_valid = time[n:m]
    # X_test = time[m:]

    X_train, X_valid, X_test = auto_data(sequenceLength=sequenceLength,tetrodeRange=tetrodeRange,num_skip=40)

    # X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    # X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1])
    # X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

    print("X_train: {}".format(X_train.shape))
    print("X_valid: {}".format(X_valid.shape))
    print("X_test: {}".format(X_test.shape))

    print("y_train: {}".format(y_train.shape))
    print("y_valid: {}".format(y_valid.shape))
    print("y_test: {}".format(y_test.shape))

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
        print("BATCH_SIZE: {}".format(batch_size))
        print("input_shape[1]: {}".format(input_shape[1]))
        print("200: {}".format(200))

        shape = [batch_size,input_shape[1],400]
        print("Shape: {}".format(shape))
        shape = tuple(shape)

        # shape = tuple(list(batch_size,input_shape[1],200))
        print(shape)

        l_in = lasagne.layers.InputLayer(shape=shape)

        print("In 1 shape: ",lasagne.layers.get_output_shape(l_in))

        l_slice_1 = lasagne.layers.SliceLayer(
            l_in,
            indices=slice(0, 200),
            axis=-1
        )

        print("Slice 1 shape: ",lasagne.layers.get_output_shape(l_slice_1))

        l_slice_2 = lasagne.layers.SliceLayer(
            l_in,
            indices=slice(200, None),
            axis=-1
        )

        print("Slice 2 shape: ",lasagne.layers.get_output_shape(l_slice_2))

        l_reshape_1= lasagne.layers.ReshapeLayer(
            l_slice_1,
            (batch_size*length,200)
            )

        l_hidden_1_1 = lasagne.layers.DenseLayer(
            l_reshape_1,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 1 1 shape: ",lasagne.layers.get_output_shape(l_hidden_1_1))

        l_code_layer_1 = lasagne.layers.DenseLayer(
            l_hidden_1_1,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            )

        print("code 1 1 shape: ",lasagne.layers.get_output_shape(l_code_layer_1))

        l_hidden_1_2 = lasagne.layers.DenseLayer(
            l_code_layer_1,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 6 1 shape: ",lasagne.layers.get_output_shape(l_hidden_1_2))

        l_almost_1 = lasagne.layers.DenseLayer(
            l_hidden_1_2,
            num_units=200,
            nonlinearity=None,
            )

        l_out_1 = lasagne.layers.ReshapeLayer(
            l_almost_1,
            (batch_size,length,200)
            )

        print("Out 1 1 shape: ",lasagne.layers.get_output_shape(l_out_1))

        l_reshape_2= lasagne.layers.ReshapeLayer(
            l_slice_2,
            (batch_size*length,200)
            )

        l_hidden_2_1 = lasagne.layers.DenseLayer(
            l_reshape_2,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 1 2 shape: ",lasagne.layers.get_output_shape(l_hidden_2_1))

        l_code_layer_2 = lasagne.layers.DenseLayer(
            l_hidden_2_1,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            )

        print("code 1 2 shape: ",lasagne.layers.get_output_shape(l_code_layer_2))

        l_hidden_2_2 = lasagne.layers.DenseLayer(
            l_code_layer_2,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 6 2 shape: ",lasagne.layers.get_output_shape(l_hidden_2_2))

        l_almost_2 = lasagne.layers.DenseLayer(
            l_hidden_2_2,
            num_units=200,
            nonlinearity=None,
            )

        l_out_2 = lasagne.layers.ReshapeLayer(
            l_almost_2,
            (batch_size,length,200)
            )

        l_out_auto = lasagne.layers.ConcatLayer(
            [l_out_1,l_out_2],
            axis=-1
            )

        print("Out 1 2 shape: ",lasagne.layers.get_output_shape(l_out_2))

        l_concat = lasagne.layers.ConcatLayer(
            [l_code_layer_1,l_code_layer_2],
            axis=-1
            )

        print("concat shape: ",lasagne.layers.get_output_shape(l_concat))

        l_hidden_3_1 = lasagne.layers.DenseLayer(
            l_concat,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        print("Hidden 1 shape: ",lasagne.layers.get_output_shape(l_hidden_3_1))

        l_dropout = lasagne.layers.DropoutLayer(
            l_hidden_3_1,
            p=0.8,
            )


        l_hidden_3_2 = lasagne.layers.DenseLayer(
            l_dropout,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        print("Hidden 2 shape: ",lasagne.layers.get_output_shape(l_hidden_3_2))

        l_reshape_3_1 = lasagne.layers.ReshapeLayer(l_hidden_3_2, (batch_size, length, num_hidden_units))

        print("Reshape_2 shape: ",lasagne.layers.get_output_shape(l_reshape_3_1))

        l_recurrent = lasagne.layers.GRULayer(
            l_reshape_3_1,
            num_hidden_units,
            grad_clipping=GRAD_CLIP,
            gradient_steps=500,
            # W_in_to_hid=lasagne.init.HeUniform(),
            # W_hid_to_hid=lasagne.init.HeUniform(),
            # nonlinearity=lasagne.nonlinearities.sigmoid
            )

        print("Recurrent shape: ",lasagne.layers.get_output_shape(l_recurrent))

        l_reshape_3_2 = lasagne.layers.ReshapeLayer(
            l_recurrent,
            (batch_size*length, num_hidden_units)
            )

        print("Reshape 3 shape: ",lasagne.layers.get_output_shape(l_reshape_3_2))

        l_recurrent_out = lasagne.layers.DenseLayer(
            l_reshape_3_2,
            num_units=output_dim,
            nonlinearity=None
            )

        print("Recurrent out shape: ",lasagne.layers.get_output_shape(l_recurrent_out))

        l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
                                            (batch_size, length, output_dim))

        print("Output shape: ",lasagne.layers.get_output_shape(l_out))

        return l_out, l_out_auto


def funcs(dataset, rec_network, auto_network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM, alpha=L2_CONSTANT, sparsity=0.01,beta=0.00005,auto_frac=10):

    """
        Method the returns the theano functions that are used in 
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables
    X1_batch = T.tensor3()
    y_batch = T.tensor3()
    l_rate = T.scalar()

    rec_layers = lasagne.layers.get_all_layers(rec_network)
    num_rec_layers = len(rec_layers)
    # print(rec_layers)

    auto_layers = lasagne.layers.get_all_layers(auto_network)

    print(auto_layers)
    print(len(auto_layers))
    print([lasagne.layers.get_output_shape(i) for i in auto_layers][11])
    # num_auto_layers1 = len(auto_layers1)

    code_layer1 = auto_layers[4]
    code_layer2 = auto_layers[11]
    # code outputs
    code_output1 = lasagne.layers.get_output(code_layer1, X1_batch)
    code_output2 = lasagne.layers.get_output(code_layer2, X1_batch) 

    # print(auto_layers1)
    # this is the cost of the network when fed throught the noisey network
    auto_train_output = lasagne.layers.get_output(auto_network, X1_batch)
    rec_train_output = lasagne.layers.get_output(rec_network, X1_batch)

    auto_cost = lasagne.objectives.squared_error(auto_train_output, X1_batch)
    rec_cost = lasagne.objectives.squared_error(rec_train_output, y_batch)

    rho_hat1 = T.mean(code_output1,axis=1)
    rho_hat2 = T.mean(code_output2,axis=1)
    L1 = T.sum(sparsity * T.log(sparsity/rho_hat1) + (1 - sparsity) * T.log((1 - sparsity)/(1 - rho_hat1)))
    L2 = T.sum(sparsity * T.log(sparsity/rho_hat2) + (1 - sparsity) * T.log((1 - sparsity)/(1 - rho_hat2)))

    auto_cost = auto_cost.mean()
    rec_cost = rec_cost.mean()
    l2 = lasagne.regularization.regularize_network_params(auto_network,lasagne.regularization.l2) + lasagne.regularization.regularize_network_params(rec_network,lasagne.regularization.l2)
    # cost = rec_cost + auto_cost  + alpha * l2 # * rec_cost * auto_frac  + beta * (L1 + L2)
    cost = rec_cost + auto_frac * auto_cost * rec_cost

    # validation cost
    valid_output = lasagne.layers.get_output(rec_network, X1_batch)
    valid_cost = lasagne.objectives.squared_error(valid_output, y_batch)
    valid_cost = valid_cost.mean()

    # test the performance of the netowork without noise
    test = lasagne.layers.get_output(rec_network, X1_batch)

    all_params = lasagne.layers.get_all_params(rec_network) + lasagne.layers.get_all_params(auto_network)
    updates = lasagne.updates.adagrad(cost, all_params, l_rate)
    train = theano.function(inputs=[X1_batch, y_batch, l_rate], outputs=cost, updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X1_batch, y_batch], outputs=valid_cost, allow_input_downcast=True)
    predict = theano.function(inputs=[X1_batch], outputs=test, allow_input_downcast=True)
    auto_cost = theano.function(inputs=[X1_batch], outputs=auto_cost, allow_input_downcast=True)
    rec_cost = theano.function(inputs=[X1_batch, y_batch], outputs=rec_cost, allow_input_downcast=True)

    return dict(
        train=train,
        valid=valid,
        predict=predict,
        auto_cost=auto_cost,
        rec_cost=rec_cost
    )

def main(tetrodeRange=[11,12]):
    """
        This is the main method that sets up the experiment
    """
    print("Loading the data...")
    dataset = load_data(tetrodeRange)
    print("Done!")
    tetrode_number = 11
    print("Tetrode number: {}, Num outputs: {}".format(tetrode_number,dataset['output_dim']))

    print("Input shape: {}".format(dataset['X_train'].shape))
    print("Output shape: {}".format(dataset['y_train'].shape))

    print("Making the model...")
    rec_network, auto_network = model(dataset['input_shape'],dataset['output_dim'])
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,rec_network, auto_network)
    print("Done!")

    if(os.path.isfile('end_network_auto_2_{}'.format(tetrode_number))):
        print("Loading old model")
        f=open('end_network_auto_2_{}'.format(tetrode_number),'r')
        all_param_values = pickle.load(f)
        f.close()
        lasagne.layers.set_all_param_values(auto_network, all_param_values)

    if(os.path.isfile('end_network_recurrent_2_{}'.format(tetrode_number))):
        print("Loading old model")
        f=open('end_network_recurrent_2_{}'.format(tetrode_number),'r')
        all_param_values = pickle.load(f)
        f.close()
        lasagne.layers.set_all_param_values(rec_network, all_param_values)

    accuracies = []
    trainvalidation = []
    learning_rate = LEARNING_RATE

    print("Begining to train the network...")
    epochsDone = 0
    increasing = 0
    k = 0
    try:
        meanTrainCost = 1000
        for i in range(NUM_EPOCHS):
            costs = []
            valid_costs = []
            auto_costs = []
            rec_costs = []

            for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
                cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end],learning_rate)
                costs.append(cost)

            for start, end in zip(range(0, dataset['num_examples_valid'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_valid'], BATCH_SIZE)):
                cost = training['valid'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
                auto_cost = training['auto_cost'](dataset['X_valid'][start:end])
                rec_cost = training['rec_cost'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
                valid_costs.append(cost)
                auto_costs.append(auto_cost)
                rec_costs.append(rec_cost)

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

            meanAutoCost = np.mean(np.asarray(auto_costs),dtype=np.float32)
            meanRecCost = np.mean(np.asarray(rec_costs),dtype=np.float32)

            print("Epoch: {}, Training cost: {}, Validation Cost: {}, learning rate: {}".format(i+1,meanTrainCost,meanValidCost,learning_rate))
            print("Rec cost: {}, Auto cost: {}".format(meanRecCost,meanAutoCost))

            if(np.isnan(meanValidCost)):
                print("Nan value")
                break

            trainvalidation.append([meanTrainCost,meanValidCost])
            # accuracies.append(accuracy)

            if(EARLY_STOPPING):
                if(len(trainvalidation) > 2):
                    if(trainvalidation[-2][1] < trainvalidation[-1][1]):
                        k += 1
                    else:
                        k = 0
                    print(k)
                    if (k == STOPPING_RANGE):
                        raise(KeyboardInterrupt)


            epochsDone = epochsDone + 1
    except KeyboardInterrupt:
        pass

    # plt.plot(trainvalidation)
    # plt.show()

    predictions = []
    cost_arrays = []
    actuals = []
    for start, end in zip(range(0, dataset['num_examples_test'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_test'], BATCH_SIZE)):
        prediction = training['predict'](dataset['X_test'][start:end])
        predictions.append(prediction)
        # accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == np.argmax(training['predict'](dataset['X_test']), axis=1))
        actuals.append(dataset['y_test'][start:end])
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

        plt.savefig('../position/test/End2_Position_test_{}.png'.format(i), bbox_inches='tight')
        plt.close()

    if(SAVE_MODEL):
        print("Saving model...")
        all_param_values = lasagne.layers.get_all_param_values(auto_network)
        f=open('end_network_auto_2_{}'.format(tetrode_number),'w')
        pickle.dump(all_param_values, f)
        f.close()
        all_param_values = lasagne.layers.get_all_param_values(rec_network)
        f=open('end_network_recurrent_2_{}'.format(tetrode_number),'w')
        pickle.dump(all_param_values, f)
        f.close()

if __name__ == '__main__':
    main()