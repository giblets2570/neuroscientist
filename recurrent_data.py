from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import json
from pos_data import getXY
import math

BASENAME = "../R2192/20140110_R2192_track1"

def getTotalInputDimension(tetrodeNumber=9,basename=BASENAME):
	"""
		Method to return the total number 
		of dimension of the input
	"""
	total = 0
	dims = []
	print type(tetrodeNumber)
	print tetrodeNumber
	for i in range(tetrodeNumber-1,16):
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
		Method that returns the times and 
	"""
	timesData = []
	header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
	# print(header)
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

def getData(tetrodeNumber=9,basename=BASENAME):
	"""
		Gets the data into the format of 
		a dictionary with the times as the 
		key and the activation over all
		tetrodes as the value (large vector)
	"""
	inputDimension, dims = getTotalInputDimension(tetrodeNumber,basename)
	currentBase = 0
	data = {}
	for k,i in enumerate(range(tetrodeNumber-1,16)):
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

def recurrentData(tetrodeNumber=9,basename=BASENAME):
	"""
		Basically convolves all the data an puts the data
		in a dictionary
	"""
	data = getData(tetrodeNumber)
	freq = 1000.0
	downData = downsampleData(data,freq=freq)
	print(len(downData))
	x, y = getXY(basename+".pos")
	print(x.shape)

	downData = gaussConv(x.shape[0],downData)

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
	conv = [gaussian(i,stepsize,stepsize/2) for i in range(2*stepsize+1)]
	# print(conv)
	for i in range(inDim)[::stepsize]:
		start = i - stepsize
		r = np.zeros(seqLen)
		for n,j in enumerate(range(start,start+2*stepsize+1)):
			if j < 0:
				continue
			if j >= inDim:
				break
			r += conv[n]*data[j]
		result.append(r)
	return np.asarray(result)


def formatData(tetrodeNumber=9,basename=BASENAME,sequenceLength=500):
	"""
		This method formats the data so it 
		can be used by the recurrent net. 
		The format is (batch_size, sequence_length,sequence_shape)

	"""

	recData = recurrentData(tetrodeNumber,basename)

	k = len(recData)
	xdim = recData[0]['activity'].shape[0]
	ydim = 2
	
	max_num_sequences = int(k/sequenceLength)
	
	# X = np.asarray([recData[i]['activity'] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,xdim))
	# y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,ydim))

	_X = np.asarray([recData[i]['activity'] for i in xrange(k)][:max_num_sequences*sequenceLength])
	_y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)][:max_num_sequences*sequenceLength])

	X = []
	i = 0
	print(_X.shape)
	while(i + sequenceLength < _X.shape[0]):
		X.append(_X[i:i+sequenceLength])
		i+=25
	X = np.asarray(X)
	print(X.shape)

	y = []
	i = 0
	print(_y.shape)
	while(i + sequenceLength < _y.shape[0]):
		y.append(_y[i:i+sequenceLength])
		i+=25
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

def test():
	m = np.zeros((20,10))
	for i in range(int(m.shape[0]/2)):
		for j in range(int(m.shape[1]/2)):
			m[i][j] = 1.0
	print(m)
	m = gaussConv(5,m)
	print(m)


if __name__=="__main__":
	
	formatData()
	

