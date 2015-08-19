from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import pickle

BASENAME = "../R2192-screening/20141001_R2192_screening"

def getData(basename=BASENAME,tetrodeRange=[9,10,11,12,13,14,15,16],freq=50.0):

	header, _ = readfile(basename+".1",[('ts','>i'),('waveform','50b')])
	print(header)

	timebase = float(re.sub("[^0-9]", "", header['timebase']))
	print(timebase)

	duration = int(re.sub("[^0-9]", "", header['duration']))
	print(duration)

	num_points=int(duration*freq)
	array_size = len(tetrodeRange)*200
	output = np.zeros((num_points,array_size))

	for tetrodeNumber in tetrodeRange:
		tetfilename = basename + "." + str(tetrodeNumber)
		print("On tetrode {}".format(tetrodeNumber))
		base = 200*(tetrodeNumber-9)
		header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
		j = 0
		for i in range(num_points):

			try:
				while data[j][0] / timebase < (i / freq):
					ind = j%4
					ind = 50*ind
					for k in range(50):
						output[ind+base+k] += data[j][1][k]
					j+=1
			except IndexError:
				pass
			if(i%10000==0):
				print("Done: {}".format(i))

	print(output.shape)
	return output

def formatData(basename=BASENAME,tetrodeRange=[9,10,11,12,13,14,15,16]):

	# f = open("activations.npy",'r')
	# activations = pickle.load(f)
	# f.close()

	# f = open("labels.npy",'r')
	# labels = pickle.load(f)
	# f.close()

	activations, labels = getData(basename,tetrodeRange)

	num_entries = activations.shape[0]
	n = int(num_entries*0.8)
	m = int(num_entries*0.9)

	print("Got the data yo")

	return activations[:n],activations[n:m],activations[m:],labels[:n],labels[n:m],labels[m:]

if __name__=="__main__":
	# activations, labels = getData(BASENAME,[9])

	out = getData()

	print(out[:20])

	print("Saving the models")

	# f=open('labels.npy','w')
	# pickle.dump(labels, f)
	# f.close()

	# print("Done 1")

	# f=open('activations.npy','w')
	# pickle.dump(activations, f)
	# f.close()
