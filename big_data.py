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
	m = 0.0
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
						# print(data[j][1][k])
						output[i][ind+k+base] += data[j][1][k]
					j+=1
			except IndexError:
				pass
			if(np.amax(output[i]) > m):
				m = np.amax(output[i])
			if(i%10000==0):
				print("Done: {}".format(i))
				# print(list(output[i]))

	print(output.shape)
	output /= m
	return output

def formatData(basename=BASENAME,tetrodeRange=[9,10,11,12,13,14,15,16]):

	activations = getData(basename,tetrodeRange)

	num_entries = activations.shape[0]
	n = int(num_entries*0.8)
	m = int(num_entries*0.9)

	print("Got the data yo")

	return activations[:n],activations[n:m],activations[m:]

if __name__=="__main__":
	# activations, labels = getData(BASENAME,[9])

	out = getData()

	print(out[:20])

	plt.plot(out[1])
	plt.show()

	print("Saving the models")

	f=open('big_input.npy','w')
	pickle.dump(out, f)
	f.close()

	# print("Done 1")

	# f=open('activations.npy','w')
	# pickle.dump(activations, f)
	# f.close()
