
# Class to make a neural net

def load_data(tetrode_number):
    """
        Get data with labels, split into training and test set.
    """

    X_train, X_valid, X_test, y_train, y_valid, y_test = formatData(tetrode_number,BASENAME,CONV)

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    X_valid = X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

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

class net(object):


	def __init__(self,dataset,network,X_type,batch_size,learning_rate,momentum,log=False,early_stopping=False):
		"""
			Initialise the parameters of the net
		"""
	
       	if network:
    		self.network = network
	       	self.batch_size = batch_size
	       	self.learning_rate = learning_rate
	       	self.momentum = momentum
	       	self.set_iter_funcs(X_type)
	       	self.log = log
	       	self.dataset = dataset
		else:
	    	self.fail = True

	def set_iter_funcs(self, X_type):
	    """
	        Method the returns the theano functions that are used in 
	        training and testing. These are the train and predict functions.
	        The predict function returns out output of the network.
	    """

	    # symbolic variables 
	    X_batch = X_type()
	    y_batch = T.matrix()

	    # this is the cost of the network when fed throught the noisey network
	    train_output = lasagne.layers.get_output(self.network, X_batch)
	    cost = lasagne.objectives.categorical_crossentropy(train_output, y_batch)
	    cost = cost.mean()

	    # validation cost
	    valid_output = lasagne.layers.get_output(self.network, X_batch)
	    valid_cost = lasagne.objectives.categorical_crossentropy(valid_output, y_batch)
	    valid_cost = valid_cost.mean()

	    # test the performance of the netowork without noise
	    test = lasagne.layers.get_output(self.network, X_batch, deterministic=True)
	    pred = T.argmax(test, axis=1)
	    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

	    all_params = lasagne.layers.get_all_params(self.network)
	    updates = lasagne.updates.nesterov_momentum(cost, all_params, self.learning_rate, self.momentum)
	    
	    train = theano.function(inputs=[X_batch, y_batch], outputs=cost, updates=updates, allow_input_downcast=True)
	    valid = theano.function(inputs=[X_batch, y_batch], outputs=valid_cost, allow_input_downcast=True)
	    predict = theano.function(inputs=[X_batch], outputs=pred, allow_input_downcast=True)

	    self.iter_funcs=dict(
	        train=train,
	        valid=valid,
	        predict=predict
	    )

	def fit(self,tetrode_number):
	    """
	        This is the main method that sets up the experiment
	    """
	    if self.fail:
	    	print("The net has not been initialised properly, exiting...")
	    	return

	    print("Loading the data...")
	    dataset = load_data(tetrode_number)
	    print("Done!")

	    print(dataset['input_shape'])

	    print("Making the model...")
	    network = model(dataset['input_shape'],dataset['output_dim'],NUM_HIDDEN_UNITS)
	    print("Done!")

	    print("Setting up the training functions...")
	    training = funcs(dataset,network)
	    print("Done!")

	    accuracies = []
	    trainvalidation = []

	    print("Begining to train the network...")
	    for i in range(NUM_EPOCHS):
	        costs = []
	        valid_costs = []

	        for start, end in zip(range(0, dataset['num_examples_train'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_train'], BATCH_SIZE)):
	            cost = training['train'](dataset['X_train'][start:end],dataset['y_train'][start:end])
	            costs.append(cost)
	        
	        for start, end in zip(range(0, dataset['num_examples_valid'], BATCH_SIZE), range(BATCH_SIZE, dataset['num_examples_valid'], BATCH_SIZE)):
	            cost = training['train'](dataset['X_valid'][start:end],dataset['y_valid'][start:end])
	            valid_costs.append(cost)

	        meanValidCost = np.mean(np.asarray(valid_costs),dtype=np.float32) 
	        meanTrainCost = np.mean(np.asarray(costs,dtype=np.float32))
	        accuracy = np.mean(np.argmax(dataset['y_test'], axis=1) == training['predict'](dataset['X_test']))

	        print("Epoch: {}, Accuracy: {}, Training cost / validation cost: {}".format(i+1,accuracy,meanTrainCost/meanValidCost))

	        trainvalidation.append(meanTrainCost/meanValidCost)

	        if(early_stopping):
	            if(len(accuracies) < 30):
	                accuracies.append(accuracy)
	            else:
	                test = [k for k in accuracies if k < accuracy]
	                if not test:
	                    print('Early stopping causing training to finish at epoch {}'.format(i+1))
	                    break
	                del accuracies[0]
	                accuracies.append(accuracy)

	    with open('trainvalidation.json',"w") as outfile:
	        outfile.write(str(trainvalidation))

	    # plt.plot(trainvalidation)
	    # plt.show()

	    if(self.log):
	        print("Logging the experiment details...")
	        log = dict(
	            TETRODE_NUMBER = tetrode_number,
	            BASENAME = BASENAME,
	            NUM_EPOCHS = NUM_EPOCHS,
	            BATCH_SIZE = BATCH_SIZE,
	            NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS,
	            LEARNING_RATE = LEARNING_RATE,
	            MOMENTUM = MOMENTUM,
	            ACCURACY = accuracies[-1],
	            NETWORK_LAYERS = [str(type(layer)) for layer in lasagne.layers.get_all_layers(network)],
	            OUTPUT_DIM = dataset['output_dim'],
	            # NETWORK_PARAMS = lasagne.layers.get_all_params_values(network)
	        )
	        now = datetime.datetime.now()
	        filename = "experiments/{}_{}_{}_NUMLAYERS_{}_OUTPUTDIM_{}".format(now,NUM_EPOCHS,NUM_HIDDEN_UNITS,len(log['NETWORK_LAYERS']),log['OUTPUT_DIM'])
	        filename = re.sub("[^A-Za-z0-9_/ ,-:]", "", filename)
	        with open(filename,"w") as outfile:
	            json.dump(log, outfile)