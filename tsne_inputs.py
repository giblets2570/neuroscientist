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

NUM_EPOCHS = 1000000
BATCH_SIZE = 400
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

EARLY_STOPPING = False
STOPPING_RANGE = 10

LOG_EXPERIMENT = True

TETRODE_NUMBER = 11

SAVE_MODEL = True

CONV = False

NUM_POINTS = 5000

DB_CONST = 1.2

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

def tsne_inputs(dataset):
	total_points = dataset['X_train'].shape[0]
	for i in range(0,total_points,NUM_POINTS):
		codes_2d = bh_sne(np.asarray(dataset['X_train'][i:i+NUM_POINTS],dtype=np.float64))
		plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=dataset['y_train_labels'][i:i+NUM_POINTS],alpha=0.8,lw=0)
		plt.savefig('../logs/plots/tsne_inputs_{}_{}.png'.format(NUM_POINTS,i), bbox_inches='tight')
		plt.close()

def cluster(dataset):
	total_points = dataset['X_train'].shape[0]
	for i in range(0,total_points,NUM_POINTS):
		codes_2d = bh_sne(np.asarray(dataset['X_train'][i:i+NUM_POINTS],dtype=np.float64))
		plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=dataset['y_train_labels'][i:i+NUM_POINTS],alpha=0.8,lw=0)
		plt.savefig('../logs/plots/tsne_inputs_{}_{}.png'.format(NUM_POINTS,i), bbox_inches='tight')
		plt.close()
		db = DBSCAN(eps=DB_CONST, min_samples=5).fit(codes_2d)
		print("Got db scan")
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

		print('Estimated number of clusters: %d' % n_clusters_)
		print("Homogeneity: %0.3f" % metrics.homogeneity_score(dataset['y_train_labels'][i:i+NUM_POINTS], labels))
		print("Completeness: %0.3f" % metrics.completeness_score(dataset['y_train_labels'][i:i+NUM_POINTS], labels))
		print("V-measure: %0.3f" % metrics.v_measure_score(dataset['y_train_labels'][i:i+NUM_POINTS], labels))
		print("Adjusted Rand Index: %0.3f"
		      % metrics.adjusted_rand_score(dataset['y_train_labels'][i:i+NUM_POINTS], labels))
		print("Adjusted Mutual Information: %0.3f"
		      % metrics.adjusted_mutual_info_score(dataset['y_train_labels'][i:i+NUM_POINTS], labels))
		print("Silhouette Coefficient: %0.3f"
		      % metrics.silhouette_score(codes_2d, labels))

		##############################################################################
		# Plot result
		# import matplotlib.pyplot as plt

		print("Plotting the db scan")

		# Black removed and is used for noise instead.
		unique_labels = set(labels)
		colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
		for k, col in zip(unique_labels, colors):
			if k == -1:
			    # Black used for noise.
				col = 'k'

			class_member_mask = (labels == k)

			xy = codes_2d[class_member_mask & core_samples_mask]
			# plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
			#          markeredgecolor='k', markersize=7)
			plt.scatter(xy[:, 0], xy[:, 1], c=col, alpha=0.8,lw=0.0)

			xy = codes_2d[class_member_mask & ~core_samples_mask]
			# plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
			#          markeredgecolor='k', markersize=4)
			plt.scatter(xy[:, 0], xy[:, 1], c=col, alpha=0.8)

		plt.title('Estimated number of clusters: %d' % n_clusters_)
		plt.savefig('../logs/plots/cluster_inputs_{}_{}_{}.png'.format(NUM_POINTS,DB_CONST,i), bbox_inches='tight')
		plt.close()


if __name__ == '__main__':
	dataset = load_data()
	cluster(dataset)