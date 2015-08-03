from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import pickle

BASENAME = "../R2192-screening/20141001_R2192_screening"

def getData(basename=BASENAME):
	output_labels_dims = []
	for tetrodeNumber in range(9,17):
		cutfilename = basename + ".clu." + str(tetrodeNumber)
		for n, i in enumerate(open(cutfilename,'r')):
			output_labels_dims.append(int(re.sub("[^0-9]", "", i)))
			if n == 0:
				break
	print(output_labels_dims)


	total_num_labels = sum(output_labels_dims)
	print(total_num_labels)

	labels = []
	for tetrodeNumber in range(9,17):
		cutfilename = basename + ".clu." + str(tetrodeNumber)
		base = tetrodeNumber - 9
		for n, i in enumerate(open(cutfilename,'r')):
			output_labels_dims.append(int(re.sub("[^0-9]", "", i)))
			if n == 0:
				continue
			datapoint = np.zeros(total_num_labels)

			datapoint[output_labels_dims[base] + int(i) - 1] = 1.0
			labels.append(datapoint)

	labels = np.asarray(labels)
	print(labels.shape)

	activations = []
	for tetrodeNumber in range(9,17):
		tetfilename = basename + "." + str(tetrodeNumber)
		print("On tetrode {}".format(tetrodeNumber))
		base = 200*(tetrodeNumber-9)
		header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
		for i in range(0,len(data),4):
			if i+3 > len(data):
				break
			m = 0
			for j in range(4):
				for k in range(50):
					val = abs(data[i+j][1][k])
					if(val > m):
						m = val
			
			con = list(np.asarray(data[i][1],dtype=np.float16)/m)+list(np.asarray(data[i+1][1],dtype=np.float16)/m)+list(np.asarray(data[i+2][1],dtype=np.float16)/m)+list(np.asarray(data[i+3][1],dtype=np.float16)/m)
			entry = np.zeros(1600)
			for i in range(base,base+200):
				entry[i] = con[i-base]
			activations.append(entry)

	activations = np.asarray(activations)

	print("Shuffling the data")

	rng_state = np.random.get_state()
	np.random.shuffle(activations)
	np.random.set_state(rng_state)
	np.random.shuffle(labels)
	print("Done Shuffling")

	return activations, labels

def formatData(basename=BASENAME):

	f = open("activations.npy",'r')
    activations = pickle.load(f)
    f.close()

    f = open("labels.npy",'r')
    labels = pickle.load(f)
    f.close()

	num_entries = activations.shape[0]
	n = int(num_entries*0.8)
	m = int(num_entries*0.9)

	print("Got the data yo")

	return activations[:n],activations[n:m],activations[m:],labels[:n],labels[n:m],labels[m:]

if __name__=="__main__":
	activations, labels = getData(BASENAME)

	f=open('activations.npy','w')
	pickle.dump(activations, f)
	f.close()

	f=open('labels.npy','w')
	pickle.dump(labels, f)
	f.close()

	print(trX.shape) 
	print(tvX.shape) 
	print(teX.shape) 
	print(trY.shape) 
	print(tvY.shape) 
	print(teY.shape)
	
