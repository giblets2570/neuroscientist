from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import json
from pos_data import getXY
import math

BASENAME = "../R2192/20140110_R2192_track1"


def getTotalInputDimension(basename=BASENAME):
	total = 0
	dims = []
	for i in range(16):
		# get the tetrode number
		cutfilename = basename + ".clu." + str(i+1)

		# get the number of output dimensions for this tetrode
		for j in open(cutfilename,'r'):
			total += int(re.sub("[^0-9]", "", j))
			dims.append(int(re.sub("[^0-9]", "", j)))
			break
	return total,dims

def getCutTimes(tetfilename,cutfilename):
	timesData = []
	header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
	labels = []
	for i,j in enumerate(open(cutfilename,'r')):
		if(i == 0):
			continue
		labels.append(int(re.sub("[^0-9]", "", j))-1)
	for i,j in zip(range(0,len(data),4),labels):
		if i+3 > len(data):
			break
		entry = {}
		entry['time'] = data[i][0]
		entry['label'] = j

		timesData.append(entry)

	return timesData

def getData(basename=BASENAME):
	inputDimension, dims = getTotalInputDimension(basename)
	currentBase = 0
	data = {}
	for i in range(16):
		dimension = dims[i]
		times = getCutTimes(basename+".{}".format(i),basename+".clu.1")
		for n,j in enumerate(times):
			if str(j['time']) in data:
				data[j['time']][currentBase + j['label']] = data[str(j['time'])][currentBase + j['label']] + 1
			else:
				data[j['time']] = np.zeros(inputDimension)
				data[j['time']][currentBase + j['label']] = 1

			# if n==10:
			# 	break
		currentBase += dimension

	return data

if __name__=="__main__":
	# inputDimension = getTotalInputDimension()
	# print(inputDimension)
	# trX, tvX, teX, trY, tvY, teY = formatData(1,BASENAME,twoD=False)

	# print(trX.shape)
	# print(trY.shape)
	# filename = BASENAME+".1"
	# cutfilename = BASENAME+".clu.1"
	# print(getCutTimes(filename,cutfilename)[:3])
	data = getData()
	# print(len(data))

	formattedData = np.zeros((69700,254))
	sigma = 0.01*1000
	for i,timestep in enumerate(xrange(0,69700)):
		milisec = (timestep+1)*1000/50.0
		for j in data:
			guassian = math.exp(-(j-milisec)**2/(2*sigma**2))/(sigma*math.sqrt(2*math.pi))
			formattedData[i]+=(data[j]*guassian)



	j = open('recurrent_data.json','w')
	f = open('recurrent_data.json','a+')
	for i in formattedData:
		f.write(i)
	# for i, thing in enumerate(data):
	# 	f.write(thing+ ' ')
	# 	f.write(str(list(data[thing])) + '\n')
	# 	# print(thing,list(data[thing]))
	# 	# if i == 4:
	# 	# 	break
	# 	print(i)
	# print("stringed the data")
	# with open('data.json','w') as f:
	# 	f.write(thing)
	# plt.plot(trX[1])
	# plt.show()

	# print(trX.shape, teX.shape, trY.shape, teY.shape)

