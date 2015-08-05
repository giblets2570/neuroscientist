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

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

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

NUM_EPOCHS = 5

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

    X_train, X_valid, X_test, y_train, y_valid, y_test = formatData(sequenceLength=2000,testing=True)

    return dict(
        X_train=X_train,
        y_train=y_train,
        # X_valid=X_valid,
        # y_valid=y_valid,
        # X_test=X_test,
        # y_test=y_test,
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

        l_hidden_2 = lasagne.layers.DenseLayer(
            l_hidden_1,
            num_units=reduced_length,
            nonlinearity=lasagne.nonlinearities.rectify
            )

        # print("Hidden 1 shape: ",lasagne.layers.get_output_shape(l_hidden_1))

        l_reshape_2 = lasagne.layers.ReshapeLayer(l_hidden_2, (batch_size, length, num_hidden_units))

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
    else:
        return 

    print("Setting up the testing functions...")
    training = funcs(dataset,network)
    print("Done!")


    print("Begining to train the network...")
    predictions = []
    actuals = []
    try:

        for i in range(NUM_EPOCHS):
            costs = []
            valid_costs = []

            for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
                cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end],LEARNING_RATE)
                costs.append(cost)

            meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
            print("Epoch: {}, Training cost: {}".format(i+1,meanTrainCost))

    except KeyboardInterrupt:
        pass

    print("Begining to test the network...")
    for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
        prediction = training['predict'](dataset['X_train'][start:end])
        predictions.append(prediction)
        # accuracy = np.mean(np.argmax(dataset['y_train'], axis=1) == np.argmax(training['predict'](dataset['X_train']), axis=1))
        actuals.append(dataset['y_train'][start:end])

    points_from = 1900

    print("Plotting the predictions")

    makeVideo(predictions[-1],actual[-1],points_from,0)


    # for i,(actual,prediction) in enumerate(zip(actuals,predictions)):
    #     prediction = np.asarray(prediction)
    #     actual = np.asarray(actual)

    #     print("Actual: {}".format(actual.shape))
    #     print("Prediction: {}".format(prediction.shape))
    #     dist = np.linalg.norm(actual-prediction)
    #     print("Distance: {}".format(dist))

    #     fig = plt.figure(1)

    #     sub1 = fig.add_subplot(121)
    #     sub2 = fig.add_subplot(122)

    #     sub1.set_title("Predicted", fontsize=16)
    #     sub2.set_title("Actual", fontsize=16)
    #     sub1.scatter(prediction[0,points_from:,0],prediction[0,points_from:,1],lw=0.0)
    #     sub1.axis([0.0,1.0,0.0,1.0])
    #     sub2.scatter(actual[0,points_from:,0],actual[0,points_from:,1],c=(1,0,0,1),lw=0.2)
    #     sub2.axis([0.0,1.0,0.0,1.0])
    #     sub1.grid(True)
    #     sub2.grid(True)

    #     fig.tight_layout()

    #     plt.savefig('../position/Position_{}.png'.format(i), bbox_inches='tight')
    #     plt.close()

def makeVideo(prediction, actual, points_from, number):

    duration = 2000 - points_from
    fps = 1

    fig = plt.figure(1)

    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)

    def make_frame(t):
        sub1.clear()
        sub2.clear()
        sub1.set_title("Predicted", fontsize=16)
        sub2.set_title("Actual", fontsize=16)

        sub1.scatter(prediction[0,points_from:points_from + 1 + t,0],prediction[0,points_from:points_from + 1 + t,1],lw=0.0)
        sub1.axis([0.0,1.0,0.0,1.0])
        sub2.scatter(actual[0,points_from:points_from + 1 + t,0],actual[0,points_from:points_from + 1 + t,1],c=(1,0,0,1),lw=0.2)
        sub2.axis([0.0,1.0,0.0,1.0])
        sub1.grid(True)
        sub2.grid(True)

        fig.tight_layout()

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration = duration)
    animation.write_gif("../position/position_{}.gif".format(number), fps=fps)

if __name__ == '__main__':
    
    main()
