from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re

BASENAME = "../R2192/20140110_R2192_track1"

def formatData(tetrodeNumber,basename):

	tetfilename = basename + "." + str(tetrodeNumber)
	cutfilename = basename + ".clu." + str(tetrodeNumber)

	def concatanateChannels():
		header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
		# print(header)
		timesData = []
		inputData = []
		for i in range(0,len(data),4):
			if i+3 > len(data):
				break
			entry = {}
			entry['time'] = data[i][0]
			m = 0
			for j in range(4):
				for k in range(50):
					val = abs(data[j][1][k])
					if(val > m):
						m = val
			
			con = list(np.asarray(data[i][1],dtype=np.float16)/m)+list(np.asarray(data[i+1][1],dtype=np.float16)/m)+list(np.asarray(data[i+2][1],dtype=np.float16)/m)+list(np.asarray(data[i+3][1],dtype=np.float16)/m)
			conData = np.asarray(con,dtype=np.float16)
			entry['data'] = conData
			timesData.append(entry)
			inputData.append(conData)

		return np.asarray(inputData),timesData

	def formatCut():
		numclusters = 0
		output = []
		for (n,i) in enumerate(open(cutfilename,'r')):
			i = re.sub("[^0-9]", "", i)
			i = int(i)
			if n == 0:
				numclusters = i
				continue

			datapoint = np.zeros(numclusters)
			datapoint[i-1] = 1
			output.append(datapoint)
		
		return np.asarray(output)

	
	def getTrainingTest():
		inp, td = concatanateChannels()
		out = formatCut()
		n = len(inp)
		n *= 0.8
		n = int(n)

		trX = inp[:n]
		teX = inp[n:]
		trY = out[:n]
		teY = out[n:]

		return trX, teX, trY, teY

	return getTrainingTest()

if __name__=="__main__":
	trX, teX, trY, teY = formatData(sys.argv[1],BASENAME)

	print(trX.shape)
	plt.plot(trX[1])
	plt.show()

	# print(trX.shape, teX.shape, trY.shape, teY.shape)
