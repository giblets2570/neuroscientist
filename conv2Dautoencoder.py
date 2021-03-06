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
BATCH_SIZE = 400
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = False
STOPPING_RANGE = 10

LOG_EXPERIMENT = True

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

def load_data(tetrode_number=TETRODE_NUMBER):
    """
        Get data with labels, split into training and test set.
    """
    print("Loading data...")
    X_train, X_valid, X_test, y_train_labels, y_valid_labels, y_test_labels = formatData(tetrode_number,BASENAME,CONV)
    print("Done!")

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])
    X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1],X_valid.shape[2])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])


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
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        labeled_test=r,
        caswells_dim = y_train_labels.shape[-1],
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_shape=X_train.shape,
        output_dim=y_train.shape[-1],
    )


def model(input_shape, output_dim, num_hidden_units,num_hidden_units_2, num_code_units, filter_size, batch_size=BATCH_SIZE):
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

        print("Input shape: ",lasagne.layers.get_output_shape(l_in))

        # print(shaped_units)
        # shaped_units = shaped_units[0]
        shaped_units = 2800

        # print(shape)

        l_conv2D_1 = lasagne.layers.Conv2DLayer(
            l_in, 
            num_filters=8,
            filter_size=filter_size, 
            stride=(1, 1), 
            border_mode="valid", 
            untie_biases=False, 
            nonlinearity=None,
            )

        print("Conv 2D shape: ",lasagne.layers.get_output_shape(l_conv2D_1))

        l_reshape_1 = lasagne.layers.ReshapeLayer(
            l_conv2D_1,
            shape=(([0], -1))
            )

        print("Reshape 1 shape: ", lasagne.layers.get_output_shape(l_reshape_1))

        l_hidden_1 = lasagne.layers.DenseLayer(
            l_reshape_1,
            num_units= num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 1 shape: ", lasagne.layers.get_output_shape(l_hidden_1))

        l_code_layer = lasagne.layers.DenseLayer(
            l_hidden_1,
            num_units=num_code_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Code layer shape: ",lasagne.layers.get_output_shape(l_code_layer))

        l_hidden_2 = lasagne.layers.DenseLayer(
            l_code_layer,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 2 shape: ",lasagne.layers.get_output_shape(l_hidden_2))

        l_hidden_3 = lasagne.layers.DenseLayer(
            l_hidden_2,
            num_units=shaped_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

        print("Hidden 3 shape: ",lasagne.layers.get_output_shape(l_hidden_3))

        l_reshape_2 = lasagne.layers.ReshapeLayer(
            l_hidden_3,
            shape=(([0],8,7,50))
            )

        print("Reshape 2 shape: ",lasagne.layers.get_output_shape(l_reshape_2))

        l_out = lasagne.layers.Conv2DLayer(
            l_reshape_2, 
            num_filters=1,
            filter_size=filter_size, 
            stride=(1, 1), 
            border_mode="valid", 
            untie_biases=False, 
            nonlinearity=None,
            )

        # print("Deconv shape: ",lasagne.layers.get_output_shape(l_deconv2D_1))

        print("Output shape: ",lasagne.layers.get_output_shape(l_out))

        return l_out

def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, sparsity=0.02, beta=0.5, momentum=MOMENTUM):

    """
        Method the returns the theano functions that are used in 
        training and testing. These are the train and predict functions.
        The predict function returns out output of the network.
    """

    # symbolic variables 
    X_batch = T.tensor4()
    y_batch = T.tensor4()

    layers = lasagne.layers.get_all_layers(network)
    num_layers = len(layers)
    print(num_layers)

    code_layer = layers[num_layers/2]

    # code output 
    code_output = lasagne.layers.get_output(code_layer, X_batch, deterministic=True)
    rho_hat = T.mean(code_output,axis=1)
    # L = T.sum(sparsity * T.log(sparsity/rho_hat) + (1 - sparsity) * T.log((1 - sparsity)/(1 - rho_hat)))
    l = T.sub(1,code_output)
    ll = T.mul(code_output,l)
    L = T.mul(4,ll)
    L = L.mean()


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

    return dict(
        train=train,
        valid=valid,
        predict=predict,
        accuracy=accuracy,
        code=code
    )

def main(tetrode_number=TETRODE_NUMBER,num_hidden_units=300,num_hidden_units_2=200,num_code_units=50):
    """
        This is the main method that sets up the experiment
    """
    print("Loading the data...")
    dataset = load_data(tetrode_number)
    print("Done!")

    print("Tetrode number: {}, Num outputs: {}".format(tetrode_number,dataset['output_dim']))

    print(dataset['input_shape'])
    print(dataset['output_dim'])
    
    print("Making the model...")
    network = model(dataset['input_shape'],dataset['output_dim'],num_hidden_units,num_hidden_units_2,num_code_units,(4,1))
    print("Done!")

    print("Setting up the training functions...")
    training = funcs(dataset,network)
    print("Done!")

    accuracies = []
    trainvalidation = []

    print("Begining to train the network...")
    epochsDone = 0
    autoencoderSameLabels = []
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

            print("Epoch: {}, Accuracy: {}, Training cost / validation cost: {}".format(i+1,accuracy,meanTrainCost/meanValidCost))

            if(np.isnan(meanTrainCost/meanValidCost)):
                print("Nan value")
                break


            # this is the test to see if the autoencoder is learning how to 
            if i%10==0:
                acs = []
                for j in range(dataset['caswells_dim']):
                    # print(dataset['labeled_test'][j].shape)
                    codes = training['code'](dataset['labeled_test'][j])
                    np.mean(np.argmax(dataset['y_test'], axis=1) == np.argmax(training['predict'](dataset['X_test']), axis=1))
                    format_codes = []
                    for code in codes:
                        # if(j==0):
                        format_codes.append(np.argmax(code))

                    prev = sorted(format_codes)[0]
                    # print(sorted(format_codes))
                    k = 0
                    same = [1]
                    for code in sorted(format_codes)[1:]:
                        if(code == prev):
                            same[k] = same[k] + 1
                        else:
                            k+=1
                            same.append(1)
                        prev = code

                    same = np.asarray(same)
                    # print(same,np.argmax(same),same[np.argmax(same)],np.sum(same))
                    label_acc = same[np.argmax(same)]*1.0/np.sum(same)
                    acs.append(label_acc)
                    print("Label: {}, Num examples: {}, Same label with autoencoder: {} ".format(j,dataset['labeled_test'][j].shape[0],label_acc))
                acs = np.asarray(acs)
                autoencoderSameLabels.append(np.mean(acs))
                print("Average agreement: {}".format(np.mean(acs)))


            if i%50 == 0:
                ran = randint(0,dataset['num_examples_test']-20)
                now = datetime.datetime.now()
                for j in range(10):
                    testing = [dataset['X_test'][ran]]
                    # print(testing[0].shape)
                    output = dataset['y_test'][ran].reshape((1, 200))[0]
                    print(output)

                    # print(np.arange(dataset['output_dim']))
                    # print(output)
                    prediction = training['predict'](testing)[0].reshape((1, 200))[0]
                    print(prediction)
                    # print(prediction)
                    # print(testing[0][0])
                    
                    code = training['code'](testing).reshape((1, 50))

                    # print(code)
                    
                    # plotting the figure

                    fig = plt.figure(1)
                    sub1 = fig.add_subplot(311)
                    sub2 = fig.add_subplot(312)
                    sub3 = fig.add_subplot(313)

                    # add titles

                    sub1.set_title('Desired output')
                    sub2.set_title('Net output')
                    sub3.set_title('Code layer output')

                    # adding x labels

                    sub1.set_xlabel('Time')
                    sub2.set_xlabel('Time')
                    sub3.set_xlabel('Code label')

                    # adding y labels

                    sub1.set_ylabel('Amplitude')
                    sub2.set_ylabel('Amplitude')
                    sub3.set_ylabel('Probability')

                    # Plotting data

                    # print(testing[0][0])
                    # inp = []
                    # for z in range(4):
                    #     inp += list(testing[0][0][z])


                    sub1.plot(output)
                    # sub1.bar(x_axis, output, width=1)
                    sub1.grid(True)

                    sub2.plot(prediction)
                    sub2.grid(True)

                    x_axis = list(np.arange(len(code[0])))

                    # sub3.plot(code[0])
                    sub3.bar(x_axis, code[0], width=1)
                    # plt.show()

                    fig.tight_layout()

                    # plt.plot(var2)
                    # fig.tight_layout()
                    plt.savefig('../logs/convAuto/fig{}_{}_{}.png'.format(i,j,now), bbox_inches='tight')
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
            NET_TYPE = "Conv auto encoder 2 hidden 1 code",
            TETRODE_NUMBER = tetrode_number,
            BASENAME = BASENAME,
            NUM_EPOCHS = epochsDone,
            BATCH_SIZE = BATCH_SIZE,
            TRAIN_VALIDATION = trainvalidation,
            LEARNING_RATE = LEARNING_RATE,
            MOMENTUM = MOMENTUM,
            SAME_LABEL_AVERAGES = autoencoderSameLabels,
            ACCURACY = accuracies,
            NETWORK_LAYERS = [str(type(layer)) for layer in lasagne.layers.get_all_layers(network)],
            OUTPUT_DIM = dataset['output_dim'],
            # NETWORK_PARAMS = lasagne.layers.get_all_params_values(network)
        )
        now = datetime.datetime.now()
        filename = "experiments/convAuto/{}_{}_{}_NUMLAYERS_{}_OUTPUTDIM_{}".format(now,NUM_EPOCHS,NUM_HIDDEN_UNITS,len(log['NETWORK_LAYERS']),log['OUTPUT_DIM'])
        filename = re.sub("[^A-Za-z0-9_/,-:]", "", filename)
        with open(filename,"w") as outfile:
            outfile.write(str(log))

if __name__ == '__main__':
    main()
    