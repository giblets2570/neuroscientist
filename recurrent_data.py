from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import json
from pos_data import getXY
import math


# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

BASENAME = "../R2192-screening/20141001_R2192_screening"

def getTotalInputDimension(tetrodeNumber=9,endTetrode=16,basename=BASENAME):
	"""
		Method to return the total number 
		of dimension of the input
	"""
	total = 0
	dims = []
	print type(tetrodeNumber)
	print tetrodeNumber
	for i in range(tetrodeNumber-1,endTetrode):
		# get the tetrode number
		print("Tetrode number: {}".format(i+1))
		cutfilename = basename + ".clu." + str(i+1)
		# get the number of output dimensions for this tetrode
		for j in open(cutfilename,'r'):
			total += int(re.sub("[^0-9]", "", j))
			dims.append(int(re.sub("[^0-9]", "", j)))
			break
	return total,dims

def getCutTimes(tetfilename,cutfilename):
	"""
		Method that returns the times and the labels of each activation.
	"""
	timesData = []
	header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
	
	labels = []
	for i,j in enumerate(open(cutfilename,'r')):
		if(i == 0):
			continue
		labels.append((int(re.sub("[^0-9]", "", j))-1))
	for i,j in zip(range(0,len(data),4),labels):
		if i+3 > len(data):
			break
		entry = {}
		entry['time'] = data[i][0] / 96000.0
		entry['label'] = j
		timesData.append(entry)
	return timesData

def getData(tetrodeNumber=9,endTetrode=16,basename=BASENAME):
	"""
		Gets the data into the format of 
		a dictionary with the times as the 
		key and the activation over all
		tetrodes as the value (large vector)
	"""
	inputDimension, dims = getTotalInputDimension(tetrodeNumber,endTetrode,basename)
	currentBase = 0
	data = {}
	for k,i in enumerate(range(tetrodeNumber-1,endTetrode)):
		dimension = dims[k]
		times = getCutTimes(basename+".{}".format(i),basename+".clu.1")
		for n,j in enumerate(times):

			if j['time'] in data:
				data[j['time']][currentBase + j['label']] = 1
			else:
				data[j['time']] = np.zeros(inputDimension)
				# for i in range(len(data[j['time']])):
				# 	data[j['time']][i] = -1
				data[j['time']][currentBase + j['label']] = 1
		currentBase += dimension
	return data

def downsampleData(data, freq=50, timeS=1394):
	"""
		Method that downsamples the data to the given frequency
	"""

	timesteps = timeS*freq

	print("Time intervals: {}".format(timesteps))

	rate = 1.0 / freq
	output = None
	first = True
	index = 0
	base = rate
	print("Making sure in order")
	for key in sorted(data):
		# print(key)
		if first:
			output = np.zeros((timesteps,len(data[key])))
			first = False
		while key > base:
			index += 1
			base += rate
		output[index] += data[key]
	return np.asarray(output)

def recurrentData(tetrodeNumber=9,endTetrode=16,basename=BASENAME):
	"""
		Basically convolves all the data an puts the data
		in a dictionary
	"""
	data = getData(tetrodeNumber,endTetrode)
	freq = 50.0
	downData = downsampleData(data,freq=freq)
	# print(len(downData))
	x, y = getXY(basename+".pos")
	print(x.shape)

	# downData = gaussConv(x.shape[0],downData)

	downData = normalizeMatrix(downData)
	print downData

	recData = []

	for i in xrange(len(downData)):
		# print i
		time = (i+1)/freq
		obj = dict(
			time=time,
			activity=downData[i],
			x=x[i],
			y=y[i]
			)
		recData.append(obj)
	return recData

def gaussConv(outDim,data):
	"""
		This is the functio that performs 
		the convolution on the input data.
		It works by having a saved conv matrix,
		and iterating over steps of the input
		while multiplying by this conv

	"""


	inDim = len(data)
	seqLen = len(data[0])
	stepsize = int(inDim/outDim)
	result = []
	conv = [gaussian(i,stepsize,stepsize/2) for i in range(stepsize+1)]
	# print(conv)
	maximum = 0.0
	for i in range(inDim)[::stepsize]:
		start = i - stepsize
		r = np.zeros(seqLen)
		for n,j in enumerate(range(start,start+stepsize+1)):
			if j < 0:
				continue
			if j >= inDim:
				break
			r += conv[n]*data[j]
		m = np.amax(r)
		if m>maximum:
			maximum = m
		result.append(r)

	return np.asarray(result)/maximum


# def formatData(tetrodeNumber=9,basename=BASENAME,sequenceLength=2000,endTetrode=16):
# 	"""
# 		This method formats the data so it 
# 		can be used by the recurrent net. 
# 		The format is (batch_size, sequence_length,sequence_shape)

# 	"""

# 	recData = recurrentData(tetrodeNumber,endTetrode,basename)

# 	k = len(recData)
# 	xdim = recData[0]['activity'].shape[0]
# 	ydim = 2
	
# 	max_num_sequences = int(k/sequenceLength)
	
# 	# X = np.asarray([recData[i]['activity'] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,xdim))
# 	# y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,ydim))

# 	_X = np.asarray([recData[i]['activity'] for i in xrange(k)][:max_num_sequences*sequenceLength])
# 	_y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)][:max_num_sequences*sequenceLength])

# 	X = []
# 	i = 0
# 	print(_X.shape)
# 	while(i + sequenceLength < _X.shape[0]):
# 		X.append(_X[i:i+sequenceLength])
# 		i+=100
# 	X = np.asarray(X)
# 	print(X.shape)

# 	y = []
# 	i = 0
# 	print(_y.shape)
# 	while(i + sequenceLength < _y.shape[0]):
# 		y.append(_y[i:i+sequenceLength])
# 		# i+=25
# 		i+=100
# 	y = np.asarray(y)
# 	print(y.shape)


# 	n = int(len(X)*0.8)
# 	m = int(len(X)*0.9)

# 	trX = np.array(X[:n],dtype=np.float32)
# 	tvX = np.array(X[n:m],dtype=np.float32)
# 	teX = np.array(X[m:],dtype=np.float32)

# 	trY = np.array(y[:n],dtype=np.float32)
# 	tvY = np.array(y[n:m],dtype=np.float32)
# 	teY = np.array(y[m:],dtype=np.float32)

# 	return trX, tvX, teX, trY, tvY, teY

def gaussian(x, mu, sig):
	output = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	if output >= 0:
		return output
	return 0.0

def normalizeMatrix(matrix):
	_max = 0.0
	for row in matrix:
		for j in row:
			if abs(j) > _max:
				_max = abs(j)
	return matrix*1.0/_max

def setZeroesNegative(m):
	for i in range(int(m.shape[0])):
		for j in range(int(m.shape[1])):
			if(m[i][j] == 0.0):
				m[i][j] = 1.0
	return m

def rate_maps(tetrodeNumber=9,endTetrode=16,basename=BASENAME):
	recData = recurrentData(tetrodeNumber,endTetrode,basename)
	k = len(recData)
	X = np.asarray([recData[i]['activity'] for i in xrange(k)])
	y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)])

	# print("k",k)
	
	num_neurons = X.shape[1]
	print(num_neurons)
	for i in range(num_neurons):
		# activations = []
		print("Neuron {}".format(i+1))

		alphas=X[:,i]
		print(alphas)
		plt.plot(alphas)
		plt.show()
		size = alphas.shape[0]
		rgba_colors = np.zeros((size,4))

		# for red the first column needs to be one
		rgba_colors[:, 2] = 1.0
		# the fourth column needs to be your alphas
		rgba_colors[:, 3] = alphas

		# print(rgba_colors)

		plt.scatter(y[:,0],y[:,1],c=rgba_colors,lw=0)

		plt.savefig('rate_maps/neuron_{}.png'.format(i), bbox_inches='tight')
		plt.close()
		# if i == 10:
		# 	break
		# plt.show()

def test():
	m = np.zeros((20,10))
	for i in range(int(m.shape[0]/2)):
		for j in range(int(m.shape[1]/2)):
			m[i][j] = 1.0
	print(m)
	m = gaussConv(5,m)
	print(m)



def organiseTetrodeData(tetrode):
	
	tetfilename = BASENAME+"."+str(tetrode)
	tetheader,tetdata = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
	print(tetheader)
	cutfilename = BASENAME+".clu."+str(tetrode)

	sample_rate = float(re.sub("[^0-9.]", "", tetheader['sample_rate']))
	timebase = float(re.sub("[^0-9.]", "", tetheader['timebase']))
	duration = int(re.sub("[^0-9.]", "", tetheader['duration']))

	data = []
	result = []
	dim = 0
	for n,j in enumerate(open(cutfilename,'r')):
		if n>0:
			label = np.zeros(dim)
			label[int(re.sub("[^0-9]", "", j))-1] = 1.0
			result.append(dict(label=label))
		else:
			dim = int(re.sub("[^0-9]", "", j))
	# print(result)
	# print(len(tetdata))
	time = 0
	activation = 0
	for n,j in enumerate(tetdata):
		if n%4==3:
			activation += list(j[1])
			activation = np.asarray(activation)
			# print(time/timebase)
			result[n/4]['time'] = time/timebase
			result[n/4]['activation'] = activation
		elif n%4==0:
			time = j[0]
			activation = list(j[1])
		else:
			activation+=list(j[1])

	# print(result[0:5])
	return duration,result

def newDownsampleData(duration,data,freq=50.0):
	print("Duration: {}".format(duration))
	i = 0
	step = 1.0/freq
	outputDim = int(duration*freq)
	activationDim = data[0]['activation'].shape[0]
	labelDim = data[0]['label'].shape[0]
	activationResult = np.zeros((outputDim,activationDim))
	labelResult = np.zeros((outputDim,labelDim))
	for entry in data:
		while((i+1)*step < entry['time']):
			i+=1
		activationResult[i] += entry['activation']
		labelResult[i] += entry['label']

	return activationResult, labelResult


def mapPosToActivations(activationResult,labelResult):
	x,y = getXY()
	result = []
	for n in xrange(x.shape[0]):
		result.append(dict(pos=[x[n],y[n]],activation=activationResult[n],label=labelResult[n]))
	return result

def ratemap(activationResult,labelResult):
	x,y = getXY()
	print(activationResult.shape)
	print(labelResult.shape)

	for i in range(labelResult[0].shape[0]):
		rgba_colors = np.zeros((x.shape[0],4))
		r = np.zeros(x.shape[0])
		rgba_colors[:, 3] = 0.2
		print("Label {}".format(i+1))
		for j in range(x.shape[0]):
			r[j] = labelResult[j][i]
			if labelResult[j][i] < 0.24:
				rgba_colors[j][3] = 0.0
			elif labelResult[j][i] < 0.55:
				rgba_colors[j][1] = 1.0
			elif labelResult[j][i] < 1:
				rgba_colors[j][0] = 1.0
				rgba_colors[j][3] = 0.8
			else:
				rgba_colors[:, 3] = 1.0
		print(np.mean(r))
		plt.scatter(x,y,c=rgba_colors,lw=0)
		plt.show()

def formatData(tetrodes=[9,10,11,12,13,14,15,16],sequenceLength=2000):

	# k = 69700
	k = 54100
	# this has to work
	totalLabel = None
	for n,tetrode in enumerate(tetrodes):
		duration, result = organiseTetrodeData(tetrode)
		activationResult, labelResult = newDownsampleData(duration,result,1000.0)
		# activationResult = gaussConv(k,activationResult)
		labelResult = gaussConv(k,labelResult)	
		# totalLabel += list(labelResult)
		print("Neuron {}".format(n))
		print(labelResult.shape)
		if n == 0:
			totalLabel = labelResult
		else:
			totalLabel = np.concatenate((totalLabel, labelResult), axis=1)

	# totalLabel = np.asarray(totalLabel)
	print(totalLabel.shape)
	_x, _y = getXY()

	xdim = totalLabel.shape[0]
	ydim = 2
	
	max_num_sequences = int(k/sequenceLength)
	
	# X = np.asarray([recData[i]['activity'] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,xdim))
	# y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,ydim))

	_X = np.asarray([totalLabel[i] for i in xrange(k)][:])
	_Y = np.asarray([[_x[i],_y[i]] for i in xrange(k)][:])

	num_skip = 20

	X = []
	i = 0
	print(_X.shape)
	while(i + sequenceLength < _X.shape[0]):
		X.append(_X[i:i+sequenceLength])
		i+=num_skip
	X = np.asarray(X)
	print(X.shape)

	y = []
	i = 0
	print(_Y.shape)
	while(i + sequenceLength < _Y.shape[0]):
		y.append(_Y[i:i+sequenceLength])
		# i+=25
		i+=num_skip
	y = np.asarray(y)
	print(y.shape)

	n = int(len(X)*0.8)
	m = int(len(X)*0.9)

	trX = np.array(X[:n],dtype=np.float32)
	tvX = np.array(X[n:m],dtype=np.float32)
	teX = np.array(X[m:],dtype=np.float32)

	trY = np.array(y[:n],dtype=np.float32)
	tvY = np.array(y[n:m],dtype=np.float32)
	teY = np.array(y[m:],dtype=np.float32)

	return trX, tvX, teX, trY, tvY, teY	

if __name__=="__main__":
	# duration, result = organiseTetrodeData(12)
	# activationResult, labelResult = newDownsampleData(duration,result,1000.0)
	# #going to test the convolution
	# print(labelResult.shape)
	# activationResult = gaussConv(69700,activationResult)
	# labelResult = gaussConv(69700,labelResult)

	# ratemap(activationResult, labelResult)

	trX, tvX, teX, trY, tvY, teY = formatData()

	# rate_maps(9,9)
	# tetfilename = BASENAME+".1"
	# header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
	# print(header)
	# cutfilename = BASENAME+".clu.1"
	# header, data = readfile(cutfilename,[('ts','>i'),('waveform','50b')])
	# print(header)
	# posfilename = BASENAME+".pos"
	# header, data = readfile(posfilename,[('ts','>i'),('pos','>8h')])
	# print(re.sub("[^0-9.]", "", header['sample_rate']))
