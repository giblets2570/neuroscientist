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
from tsne import bh_sne
from itertools import cycle

from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import DPGMM
from sklearn.manifold import TSNE
from pos_data import getXY
# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

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

NUM_EPOCHS = 50
BATCH_SIZE = 400
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = False
STOPPING_RANGE = 10

LOG_EXPERIMENT = True
TETRODE_NUMBER = 11
SAVE_MODEL = False
CONV = False
NUM_POINTS = 100000

L2_CONSTANT = 0.0000

MODEL_FILENAME = "auto_models/deep/auto_network_"

def load_data(tetrode_number=TETRODE_NUMBER):
    """
        Get data with labels, split into training and test set.
    """
    print("Loading data...")
    data, timed_activations, labels = formatData(tetrode_number,BASENAME,CONV,timed=True)
    print(len(timed_activations))
    x, y = getXY()
    print("Done!")

    return dict(
        data=data,
        labels=[np.argmax(y) for y in labels],
        timed_activations=timed_activations,
        x=x,
        y=y,
        freq=50.0
    )

def model(input_shape, output_dim, num_hidden_units,num_hidden_units_2,num_hidden_units_3, num_code_units, batch_size=BATCH_SIZE):
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

    l_code_layer = lasagne.layers.DenseLayer(
        l_hidden_1,
        num_units=num_code_units,
        nonlinearity=lasagne.nonlinearities.sigmoid,
        )

    l_hidden_6 = lasagne.layers.DenseLayer(
        l_code_layer,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
        )

    l_out = lasagne.layers.DenseLayer(
        l_hidden_6,
        num_units=output_dim,
        nonlinearity=None,
        )

    return l_out

def funcs(dataset, network, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, sparsity=0.02, beta=0.1, momentum=MOMENTUM, alpha=L2_CONSTANT):

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

    code_layer = layers[num_layers/2]
    activations_2_layer = layers[num_layers/2 - 1]
    activations_1_layer = layers[num_layers/2 - 2]

    # code output
    code_output = lasagne.layers.get_output(code_layer, X_batch, deterministic=True)

    # l = T.sub(1,code_output)
    # ll = T.mul(code_output,l)
    # L = T.mul(4,ll)
    # L = L.mean()

    rho_hat = T.mean(code_output,axis=1)
    L = T.sum(sparsity * T.log(sparsity/rho_hat) + (1 - sparsity) * T.log((1 - sparsity)/(1 - rho_hat)))

    # reg = 0.0001*lasagne.regularization.l2(network)
    # this is the cost of the network when fed throught the noisey network
    train_output = lasagne.layers.get_output(network, X_batch)
    cost = lasagne.objectives.mse(train_output, y_batch) 
    l2 = lasagne.regularization.l2(X_batch)
    cost = cost.mean() + beta * L + alpha * l2

    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate, momentum)



    # code and activation outputs

    activations_1_output = lasagne.layers.get_output(activations_1_layer, X_batch, deterministic=True)
    activations_2_output = lasagne.layers.get_output(activations_2_layer, X_batch, deterministic=True)

    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
    code = theano.function(inputs=[X_batch], outputs=code_output, allow_input_downcast=True)
    activations_1 = theano.function(inputs=[X_batch], outputs=activations_1_output, allow_input_downcast=True)
    activations_2 = theano.function(inputs=[X_batch], outputs=activations_2_output, allow_input_downcast=True)

    return dict(
        train=train,
        code=code,
        activations_1=activations_1,
        activations_2=activations_2
    )

def makeVideo(X_2d,dataset):

    labels = dataset['labels']
    timed_activations = dataset['timed_activations']
    duration = X_2d.shape[0]

    fps = 5

    fig, ax = plt.subplots(1, figsize=(4, 4), facecolor='white')
    fig.subplots_adjust(left=0, right=1, bottom=0)

    def make_frame(t):
        ax.clear()
        print(t)

        ax.set_title("Activations", fontsize=16)
        ax.scatter(X_2d[:,0],X_2d[:,1],alpha=0.1,lw=0.0)

        ax.scatter(X_2d[t*fps:t*fps+1,0],X_2d[t*fps:t*fps+1,1],alpha=1,lw=0.0)

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration = duration)
    animation.write_gif("code_tnse.gif", fps=fps)

def main(tetrode_number=TETRODE_NUMBER,num_hidden_units=100,num_hidden_units_2=300,num_hidden_units_3=200,num_code_units=50):
    """
        This is the main method that sets up the experiment
    """

    print("Making the model...")
    network = model((None,200),200,num_hidden_units,num_hidden_units_2,num_hidden_units_3,num_code_units)
    print("Done!")

    for tetrode_number in [9,10,11,12,13,14,15,16]:

        print("Loading the model parameters from {}".format(MODEL_FILENAME+str(tetrode_number)))
        f = open(MODEL_FILENAME+str(tetrode_number),'r')
        all_param_values = pickle.load(f)
        f.close()
        # print(all_param_values)
        lasagne.layers.set_all_param_values(network, all_param_values)

        print("Loading the data...")
        dataset = load_data(tetrode_number)
        print("Done!")

        print(dataset['data'].shape)

        print("Setting up the training functions...")
        training = funcs(dataset,network)
        print("Done!")

        for i in range(NUM_EPOCHS):
            costs = []

            for start, end in zip(range(0, dataset['data'].shape[0], BATCH_SIZE), range(BATCH_SIZE, dataset['data'].shape[0], BATCH_SIZE)):
                cost = training['train'](dataset['data'][start:end],dataset['data'][start:end])
                costs.append(cost)

            meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
            # accuracy = training['accuracy'](dataset['X_test'],dataset['y_test'])

            print("Epoch: {}, Training cost: {}".format(i+1,meanTrainCost))
        # NUM_POINTS = 5000
        codes = training['code'](dataset['data'][0:NUM_POINTS])

        col = [np.argmax(code) for code in codes]
        num_col = len(list(set(col)))
        already = {}
        argmax_labels = []
        n = 0
        for c in col:
            if not c in already:
                already[c] = n
                n+=1
            argmax_labels.append(already[c])

        print(len(already))

        f=open('dbscan_labels/deep/arg_tetrode_{}.npy'.format(tetrode_number),'w')
        pickle.dump(col, f)
        f.close()

        # y = set(list(d.predict(dataset['data'][0:NUM_POINTS])))

        # print(y)

        # activations_1 = training['activations_1'](dataset['data'][0:NUM_POINTS])
        # activations_2 = training['activations_2'](dataset['data'][0:NUM_POINTS])
        # codes = training['code'](dataset['data'][0:NUM_POINTS])
        # # print(codes.shape)
        # # codes_2d = bh_sne(codes)

        # for k in range(3):
        #     print(k)

        #     codes_2d = bh_sne(np.asarray(codes[:(k+1)*12000],dtype=np.float64))

        #     # d = DPGMM(n_components=10, covariance_type='full')
        #     d = DPGMM(n_components=15, covariance_type='full')

        #     d.fit(codes_2d[:(k+1)*12000])

        #     hdp = d.predict_proba(codes_2d[:(k+1)*12000])

        #     hdp_1d = [np.argmax(z) for z in hdp]

        #     print(set(list(hdp_1d)))

        #     plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=hdp_1d, alpha=0.8,lw=0)
        #     plt.savefig('dbscan_labels/deep/hdp_{}_{}.png'.format(tetrode_number,k), bbox_inches='tight')
        #     plt.close()

        #     # m = TSNE(n_components=2, random_state=0)

        #     # codes_2d = m.fit_transform(codes[:NUM_POINTS])
        #     # activations_1_2d = bh_sne(activations_1)
        #     # activations_2_2d = bh_sne(activations_2)

        #     plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=dataset['labels'][0:NUM_POINTS][:(k+1)*12000],alpha=0.8,lw=0)
        #     plt.savefig('dbscan_labels/deep/tsne_codes_{}_{}.png'.format(tetrode_number,k), bbox_inches='tight')
        #     plt.close()

        #     # This is where the code for the video will go
        #     ##############################################################################
        #     # Compute DBSCAN
        #     db = None
        #     core_samples_mask = None
        #     labels = None

        #     num_labels = 0
        #     eps=1.0
        #     while(num_labels < 10):
        #         db = DBSCAN(eps=eps, min_samples=10).fit(codes_2d)
        #         core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #         core_samples_mask[db.core_sample_indices_] = True
        #         labels = db.labels_
        #         num_labels = np.amax(labels)
        #         eps -= 0.1

        #     print("Num learned labels: {}".format(num_labels))

        #     plt.title('Estimated number of clusters: {}'.format(np.amax(labels)))
        #     plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=labels[0:NUM_POINTS][:(k+1)*12000],lw=0)
        #     plt.savefig('dbscan_labels/deep/dbscan_codes_{}_{}.png'.format(tetrode_number,k), bbox_inches='tight')
        #     plt.close()

        #     # f=open('dbscan_labels/deep/tetrode_{}.npy'.format(tetrode_number),'w')
        #     # pickle.dump(labels, f)
        #     # f.close()

        # codes_2d = bh_sne(np.asarray(codes,dtype=np.float64))


        # plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=col, alpha=0.8,lw=0)
        # plt.savefig('dbscan_labels/deep/argmax_{}.png'.format(tetrode_number), bbox_inches='tight')
        # plt.close()

        # # d = DPGMM(n_components=10, covariance_type='full')
        # d = DPGMM(n_components=15)

        # d.fit(codes_2d)

        # hdp = d.predict_proba(codes_2d)

        # hdp_1d = [np.argmax(z) for z in hdp]

        # print(set(list(hdp_1d)))

        # plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=hdp_1d, alpha=0.8,lw=0)
        # plt.savefig('dbscan_labels/deep/hdp_{}.png'.format(tetrode_number), bbox_inches='tight')
        # plt.close()

        # # m = TSNE(n_components=2, random_state=0)

        # # codes_2d = m.fit_transform(codes[:NUM_POINTS])
        # # activations_1_2d = bh_sne(activations_1)
        # # activations_2_2d = bh_sne(activations_2)

        # plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=dataset['labels'][0:NUM_POINTS],alpha=0.8,lw=0)
        # plt.savefig('dbscan_labels/deep/tsne_codes_{}.png'.format(tetrode_number), bbox_inches='tight')
        # plt.close()

        # # This is where the code for the video will go
        # ##############################################################################
        # # Compute DBSCAN
        # db = None
        # core_samples_mask = None
        # labels = None

        # num_labels = 0
        # eps=1.0
        # diff = 0.1
        # # while(num_labels < 10 or num_labels>25):
        # print("Getting the labels: {}, eps: {}".format(num_labels,eps))
        # db = DBSCAN(eps=eps, min_samples=15).fit(codes_2d)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_
        # num_labels = np.amax(labels)
        #     # if(eps <= 2*diff):
        #     #     diff *= 0.1
        #     # if(num_labels < 10):
        #     #     eps -= diff
        #     # if(num_labels > 25):
        #     #     eps += 0.5*diff

        # print("Num learned labels: {}".format(num_labels))

        # f=open('dbscan_labels/deep/tetrode_{}.npy'.format(tetrode_number),'w')
        # pickle.dump(labels, f)
        # f.close()

        # plt.title('Estimated number of clusters: {}'.format(np.amax(labels)+1))
        # plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=labels[0:NUM_POINTS],lw=0)
        # plt.savefig('dbscan_labels/deep/dbscan_tsne_{}.png'.format(tetrode_number), bbox_inches='tight')
        # plt.close()

        # num_labels = 0
        # eps=0.1
        # diff = 0.01
        # while(num_labels < 10):
        #     print("Getting the labels: {}, eps: {}".format(num_labels,eps))
        #     db = DBSCAN(eps=eps, min_samples=10).fit(codes[:15000])
        #     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #     core_samples_mask[db.core_sample_indices_] = True
        #     labels = db.labels_
        #     num_labels = np.amax(labels)
        #     if(eps <= 2*diff):
        #         diff *= 0.1
        #     eps -= diff

        # print("Num learned labels: {}".format(num_labels))

        # plt.title('Estimated number of clusters: {}'.format(np.amax(labels)))
        # plt.scatter(codes_2d[:15000, 0], codes_2d[:15000, 1], c=labels[0:NUM_POINTS][:15000],lw=0)
        # plt.savefig('dbscan_labels/deep/dbscan_codes_{}.png'.format(tetrode_number), bbox_inches='tight')
        # plt.close()

if __name__ == '__main__':
    main()