from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer, RecurrentLayer, ReshapeLayer, DenseLayer
from lasagne.nonlinearities import tanh
from lasagne.init import Normal
from lasagne.objectives import Objective

from neuralnilm.layers import MixtureDensityLayer
from neuralnilm.objectives import mdn_nll
from neuralnilm.utils import sfloatX
from neuralnilm.updates import anneal_learning_rate
from neuralnilm.plot import MDNPlotter


# Number of units in the hidden (recurrent) layer
N_HIDDEN_LAYERS = 2
N_UNITS_PER_LAYER = 25
N_COMPONENTS = 2
# Number of training sequences in each batch
N_SEQ_PER_BATCH = 16
SEQ_LENGTH = 256
SHAPE = (N_SEQ_PER_BATCH, SEQ_LENGTH, 1)
# SGD learning rate
INITIAL_LEARNING_RATE = sfloatX(5e-4)
# LEARNING_RATE_NORMALISER = 5
learning_rate = theano.shared(INITIAL_LEARNING_RATE, name='learning_rate')
LEARNING_RATE_CHANGES = {
    500: 1e-04, 
    1000: 5e-05, 
    2000: 1e-05, 
    3000: 5e-06,
    4000: 1e-06,
    10000: 5e-07,
    50000: 1e-07
}

# Number of iterations to train the net
N_ITERATIONS = 100
VALIDATE = False
VALIDATION_INTERVAL = 100

np.random.seed(42)


def gen_data():
    '''
    Generate toy data.
    :returns:
        - X : np.ndarray, shape=SHAPE
            Input sequence
        - t : np.ndarray, shape=SHAPE
            Target sequence
    '''
    NOISE_MAGNITUDE = 0.1
    PULSE_WIDTH = 10
    START = 100
    STOP = 250
    ON = 1.0
    OFF = 0.0
    def noise():
        return floatX(np.random.uniform(
            low=-NOISE_MAGNITUDE, high=NOISE_MAGNITUDE, size=SHAPE))

    t = np.zeros(shape=SHAPE, dtype=np.float32) + OFF
    X = np.zeros(shape=SHAPE, dtype=np.float32) + OFF
    X[:,START:STOP,:] = ON

    for batch_i in range(N_SEQ_PER_BATCH):
        if np.random.binomial(n=1, p=0.5):
            for pulse_start in range(START, STOP, PULSE_WIDTH*2):
                pulse_end = pulse_start + PULSE_WIDTH
                X[batch_i, pulse_start:pulse_end, 0] = OFF
            t[batch_i, :, 0] = X[batch_i, :, 0].copy()
    X += noise()
    return X, t


X_val, t_val = gen_data()

# Configure layers
layers = [InputLayer(shape=SHAPE)]
for i in range(N_HIDDEN_LAYERS):
    layer = RecurrentLayer(
        layers[-1], N_UNITS_PER_LAYER, nonlinearity=tanh, 
        W_in_to_hid=Normal(std=1.0/np.sqrt(layers[-1].get_output_shape()[-1])),
        gradient_steps=100)
    layers.append(layer)
layers.append(ReshapeLayer(layers[-1], (N_SEQ_PER_BATCH * SEQ_LENGTH, N_UNITS_PER_LAYER)))
layers.append(
    MixtureDensityLayer(
        layers[-1], 
        num_units=t_val.shape[-1], 
        num_components=N_COMPONENTS,
        min_sigma=0
    )
)

print("Total parameters: {}".format(
    sum([p.get_value().size 
         for p in lasagne.layers.get_all_params(layers[-1])])))

X = T.tensor3('X')
t = T.matrix('t')

# add test values
X.tag.test_value = floatX(np.random.rand(*SHAPE))
t.tag.test_value = floatX(np.random.rand(N_SEQ_PER_BATCH * SEQ_LENGTH, 1))

objective = Objective(layers[-1], loss_function=mdn_nll)
loss = objective.get_loss(X, t)

all_params = lasagne.layers.get_all_params(layers[-1])
updates = lasagne.updates.momentum(loss, all_params, learning_rate)

# Theano functions for training, getting output, and computing loss
print("Compiling Theano functions...")
train = theano.function([X, t], loss, updates=updates)
y_pred = theano.function([X], layers[-1].get_output(X))
compute_loss = theano.function([X, t], loss)
print("Done compiling Theano functions.")

# Train the net
costs = []
t_val = t_val.reshape((N_SEQ_PER_BATCH * SEQ_LENGTH, 1))
time_0 = time_validation = time.time()

print("Starting training...")
for n in range(N_ITERATIONS):
    X, t = gen_data()
    t = t.reshape((N_SEQ_PER_BATCH * SEQ_LENGTH, 1))
    costs.append(train(X, t))
    if not n % VALIDATION_INTERVAL:
        if VALIDATE:
            cost_val = compute_loss(X_val, t_val)
        print("*********** ITERATION", len(costs), "***********")
        if VALIDATE:
            print("    Validation cost     = {}".format(cost_val))
        print("    Training costs (for last", VALIDATION_INTERVAL, "iterations):")
        recent_costs = np.array(costs[-VALIDATION_INTERVAL:])
        print("       min = {}".format(recent_costs.min()))
        print("      mean = {}".format(recent_costs.mean()))
        print("       max = {}".format(recent_costs.max()))
        print("    Time since last validation = {:.3f}s; total time = {:.1f}s"
              .format(time.time() - time_validation, time.time() - time_0))
        print("        LR = {}".format(learning_rate.get_value()))
        time_validation = time.time()
        # learning_rate.set_value(anneal_learning_rate(
        #     INITIAL_LEARNING_RATE, LEARNING_RATE_NORMALISER, n))
        if n in LEARNING_RATE_CHANGES:
            new_learning_rate = LEARNING_RATE_CHANGES[n]
            print("######## NEW LEARNING RATE", new_learning_rate)
            learning_rate.set_value(sfloatX(new_learning_rate))


# Plot costs
ax = plt.gca()
ax.plot(costs)
ax.set_title("Training costs")
ax.grid(True)
plt.show()

# Plot means
output = y_pred(X_val)

plotter = MDNPlotter(seq_length=SEQ_LENGTH)
plotter.save = False
_X_val, _t_val, _output = plotter._process(X_val, t_val, output, target_shape=SHAPE)
fig, axes = plotter.create_estimates_fig(_X_val, _t_val, _output)
plt.show()

"""
after 5000 iterations:
  -1.42 with all biases activated
  -1.13 with no biases
  -1.087 with bias for my but not for sigma and mixing.  It does go down to -1.57
   after a total of 6000 iterations but then the cost starts bouncing around,
   I guess because sigma gets too small?
after 10,000 iterations:
  -1.7 with min_sigma=1E-6.  Got down to -1.83 but spiked up to 4.93 after 9700 iterations
  takes 945.9s in total before using const
  takes 942.2s using const
"""