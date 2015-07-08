from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import json
from pos_data import getXY
import math

BASENAME = "../R2192/20140110_R2192_track1"

def getTotalInputDimension(tetrodeNumber=11,basename=BASENAME):
	"""
		Method to return the total number 
		of dimension of the input
	"""
	total = 0
	dims = []
	print type(tetrodeNumber)
	print tetrodeNumber
	for i in range(tetrodeNumber-1,tetrodeNumber):
		# get the tetrode number
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
	print(header)
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

def getData(tetrodeNumber=11,basename=BASENAME):
	inputDimension, dims = getTotalInputDimension(tetrodeNumber,basename)
	currentBase = 0
	data = {}
	for k,i in enumerate(range(tetrodeNumber-1,tetrodeNumber)):
		dimension = dims[k]
		times = getCutTimes(basename+".{}".format(i),basename+".clu.1")
		for n,j in enumerate(times):

			if str(j['time']) in data:
				data[j['time']][currentBase + j['label']] = data[str(j['time'])][currentBase + j['label']] + 1
			else:
				data[j['time']] = np.zeros(inputDimension)
				data[j['time']][currentBase + j['label']] = 1
		currentBase += dimension
	return data

def downsampleData(data, freq=50, timeS=1394):
	"""
		Method that downsamples the data to the given frequency
	"""
	timesteps = timeS*freq
	rate = 1.0 / freq
	output = None
	first = True
	index = 0
	base = rate
	for key in sorted(data):
		if first:
			output = np.zeros((timesteps,len(data[key])))
			first = False
		if key > base:
			index += 1
			base += rate
		output[index] += data[key]
	return np.asarray(output)

def gaussianMatrix(outDim,inDim):
	out = []
	for i in range(inDim):
		out.append([gaussian((k+0.5)*inDim/outDim,i,5) for k in range(outDim)])
		
		print('{} done!'.format(i))
	return np.asarray(out).T

def firstRecurrentData(tetrodeNumber=11,basename=BASENAME):
	"""
		Basically convolves all the data
	"""
	data = getData(tetrodeNumber)
	freq = 1000.0
	downData = downsampleData(data,freq=freq)

	x, y = getXY(basename+".pos")
	print(x.shape)

	downData = gaussConv(x.shape[0],downData)

	recData = []

	for i in xrange(len(downData)):
		time = i/freq
		obj = dict(
			time=time,
			activity=downData[i],
			x=x[i],
			y=y[i]
			)
		recData.append(obj)
	return recData

def linearConv(outDim,data):
	inDim = len(data)
	seqLen = len(data[0])
	stepsize = int(inDim/outDim)
	result = []
	for i in range(inDim)[::stepsize]:
		start = i - stepsize
		r = np.zeros(seqLen)
		for j in range(start,start+2*stepsize+2):
			if j < 0 or j >= inDim:
				continue
			if j <= i:
				r += abs(j-i)/stepsize*data[i]
			else:
				r += abs(i-j)/stepsize*data[i]
		result.append(r)
	return np.asarray(result)

def gaussConv(outDim,data):
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
			if j < 0 or j >= inDim:
				continue
			r += conv[n]*data[j]
		result.append(r)
	return np.asarray(result)


def formatData(tetrodeNumber=11,basename=BASENAME,sequenceLength=500):
	recData = firstRecurrentData(tetrodeNumber,basename)

	k = len(recData)
	xdim = recData[0]['activity'].shape[0]
	ydim = 2
	
	max_num_sequences = int(k/sequenceLength)
	
	X = np.asarray([recData[i]['activity'] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,xdim))
	y = np.asarray([[recData[i]['x'],recData[i]['y']] for i in xrange(k)][:max_num_sequences*sequenceLength]).reshape((max_num_sequences, sequenceLength,ydim))

	n = int(len(X)*0.8)
	m = int(len(X)*0.9)

	trX = np.array(X[:n],dtype=np.float16)
	tvX = np.array(X[n:m],dtype=np.float16)
	teX = np.array(X[m:],dtype=np.float16)

	trY = np.array(y[:n],dtype=np.float16)
	tvY = np.array(y[n:m],dtype=np.float16)
	teY = np.array(y[m:],dtype=np.float16)

	return trX, tvX, teX, trY, tvY, teY

def gaussian(x, mu, sig):
	output = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	if output >= 0:
		return output
	return 0.0

if __name__=="__main__":
	
	# out = convolvedData()	

	# print(out.shape)

	# # r = [gaussian(i,35,17) for i in range(71)]

	# # print r

	# # plt.plot(r)

	# # plt.show()

	trX, tvX, teX, trY, tvY, teY = formatData()

	print(trX[0][0])

	# np.save('convolved_data.npy', out)

	# data = np.random.rand(30,2)

	# l = linearConv(10,data)

	# print(l.shape)
