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
from data import formatData
import json
import re
import math
import matplotlib.pyplot as plt

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

NUM_EPOCHS = 1000000
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = False
STOPPING_RANGE = 10

LOG_EXPERIMENT = False

TETRODE_NUMBER = 11

CONV = True

class DimshuffleLayer(lasagne.layers.Layer):
    def __init__(self, input_layer, pattern):
        super(DimshuffleLayer, self).__init__(input_layer)
        self.pattern = pattern

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(self.pattern)

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    X_train, X_valid, X_test, y_train, y_valid, y_test = formatData(tetrode_number,BASENAME,CONV)

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])
    X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1],X_valid.shape[2])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])

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

def model(input_shape, output_dim, filter_size, pool_size, filter_size_2, pool_size_2, num_hidden_units_1, num_hidden_units_2, num_hidden_units_3, num_hidden_units_4,num_hidden_units_5, batch_size=BATCH_SIZE):
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
        print(filter_size, pool_size, filter_size_2, pool_size_2, num_hidden_units_1, num_hidden_units_2, num_hidden_units_3,num_hidden_units_4,num_hidden_units_5)
        l_in = lasagne.layers.InputLayer(shape=shape)

        l_conv2D_1 = lasagne.layers.Conv2DLayer(
            l_in, 
            num_filters=32,
            filter_size=filter_size, 
            stride=(1, 1), 
            border_mode="valid", 
            untie_biases=False, 
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        # l_pool2D_1 = lasagne.layers.MaxPool2DLayer(
        #     l_conv2D_1, 
        #     pool_size=pool_size, 
        #     stride=None, 
        #     pad=0, 
        #     ignore_border=False,
        # )

        # l_pool2D_1 = lasagne.layers.FeaturePoolLayer(
        #     l_conv2D_1, 
        #     pool_size=pool_size, 
        # )

        # l_conv2D_2 = lasagne.layers.Conv2DLayer(
        #     l_conv2D_1, 
        #     num_filters=32,
        #     filter_size=filter_size_2, 
        #     stride=(1, 1), 
        #     border_mode="valid", 
        #     untie_biases=False, 
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     )

        # l_pool2D_2 = lasagne.layers.MaxPool2DLayer(
        #     l_conv2D_2, 
        #     pool_size=pool_size_2, 
        #     stride=None, 
        #     pad=0, 
        #     ignore_border=False,
        # )

        # # l_pool2D_2 = lasagne.layers.FeaturePoolLayer(
        # #     l_conv2D_2, 
        # #     pool_size=2, 
        # # )

        l_hidden_1 = lasagne.layers.DenseLayer(
            l_conv2D_1,
            num_units=num_hidden_units_1,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_hidden_2 = lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units_2,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_hidden_3 = lasagne.layers.DenseLayer(
            l_hidden_2,
            num_units=num_hidden_units_3,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_hidden_4 = lasagne.layers.DenseLayer(
            l_hidden_3,
            num_units=num_hidden_units_4,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_hidden_5 = lasagne.layers.DenseLayer(
            l_hidden_4,
            num_units=num_hidden_units_5,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_out = lasagne.layers.DenseLayer(
            l_hidden_5,
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
    X_batch = T.tensor4()
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

def main(filter_size,pool_size,filter_size_2,pool_size_2,num_hidden_units_1, num_hidden_units_2, num_hidden_units_3,num_hidden_units_4, num_hidden_units_5, tetrode_number=TETRODE_NUMBER):
    """
        This is the main method that sets up the experiment
    """
    print("Loading the data...")
    dataset = load_data(tetrode_number)
    print("Done!")

    print("Tetrode number: {}, Num outputs: {}".format(tetrode_number,dataset['output_dim']))

    print(dataset['input_shape'])
    
    print("Making the model...")
    network = model(dataset['input_shape'],dataset['output_dim'],filter_size,pool_size,filter_size_2,pool_size_2,num_hidden_units_1, num_hidden_units_2, num_hidden_units_3,num_hidden_units_4, num_hidden_units_5)
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
            accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == np.argmax(training['predict'](dataset['X_test']), axis=1))

            print("Epoch: {}, Accuracy: {}, Training cost / validation cost: {}".format(i+1,accuracy,meanTrainCost/meanValidCost))

            if(np.isnan(meanTrainCost/meanValidCost)):
                print("Nan value")
                break

            if i%50 == 0:
                ran = randint(0,dataset['num_examples_test']-20)
                for j in range(10):
                    testing = [dataset['X_test'][ran]]
                    print(testing[0].shape)
                    output = dataset['y_test'][ran]
                    print(np.arange(dataset['output_dim']))
                    print(output)
                    prediction = training['predict'](testing)[0]
                    print(prediction)
                    # print(testing[0][0])
                    
                    # plotting the figure

                    fig = plt.figure(1)
                    sub1 = fig.add_subplot(311)
                    sub2 = fig.add_subplot(312)
                    sub3 = fig.add_subplot(313)

                    # add titles

                    sub1.set_title('Desired output')
                    sub2.set_title('Net output')
                    sub3.set_title('Four channals of tetrode concatenated')

                    # adding x labels

                    sub1.set_xlabel('Clusters')
                    sub2.set_xlabel('Clusters')
                    sub3.set_xlabel('Time')

                    # adding y labels

                    sub1.set_ylabel('Output')
                    sub2.set_ylabel('Probability')
                    sub3.set_ylabel('Amplitude')

                    # Plotting data

                    print(testing[0][0])
                    inp = []
                    for z in range(4):
                        inp += list(testing[0][0][z])


                    x_axis = list(np.arange(dataset['output_dim']))

                    # sub1.plot(output, 'ro')
                    sub1.bar(x_axis, output, width=1)
                    sub1.grid(True)

                    # sub2.plot(prediction, 'ro')
                    sub2.bar(x_axis, prediction, width=1)
                    sub2.grid(True)

                    sub3.plot(inp)
                    # plt.show()

                    fig.tight_layout()

                    # plt.plot(var2)
                    # fig.tight_layout()
                    plt.savefig('./logs/fig{}_{}.png'.format(i,j), bbox_inches='tight')
                    plt.close()
                    
                    ran += 1
                # break


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

    except KeyboardInterrupt:
        pass

    # plt.plot(trainvalidation)
    # plt.show()

    if(LOG_EXPERIMENT):
        print("Logging the experiment details...")
        log = dict(
            NET_TYPE = "2D conv, filter_size {}, pool_size {},filter_size_2 {}, pool_size_2 {}, 3 hidden".format(filter_size,pool_size,filter_size_2,pool_size_2),
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
        filename = "experiments/conv2D/{}_{}_{}_NUMLAYERS_{}_OUTPUTDIM_{}".format(now,NUM_EPOCHS,NUM_HIDDEN_UNITS,len(log['NETWORK_LAYERS']),log['OUTPUT_DIM'])
        filename = re.sub("[^A-Za-z0-9_/ ,-:]", "", filename)
        with open(filename,"w") as outfile:
            outfile.write(str(log))



if __name__ == '__main__':
    # for i in range(16):
    #     main(1+i)
    # filter_sizes = [(4,1),(4,3),(4,5),(4,7)]

    # pool_sizes = [(1,2),(1,4),(1,8)]
    filter_size = (4,1)
    pool_size = (1,2)
    filter_size_2 = (1,7)
    pool_size_2 = (1,2)
    num_hidden_units_1 = 800
    num_hidden_units_2 = 500
    num_hidden_units_3 = 300
    num_hidden_units_4 = 200
    num_hidden_units_5 = 100
    main(filter_size,pool_size,filter_size_2,pool_size_2,num_hidden_units_1,num_hidden_units_2,num_hidden_units_3,num_hidden_units_4,num_hidden_units_5)
