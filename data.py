from reading_and_viewing_data.readDACQfile import _readFile as readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import re

BASENAME = "../R2192-screening/20141001_R2192_screening"

def formatData(tetrodeNumber,basename,twoD=False,timed=False):

	tetfilename = basename + "." + str(tetrodeNumber)
	cutfilename = basename + ".clu." + str(tetrodeNumber)

	def concatanateChannels():
		header, data = readfile(tetfilename,[('ts','>i'),('waveform','50b')])
		print(header)
		timesData = []
		inputData = []
		for i in range(0,len(data),4):
			if i+3 > len(data):
				break
			entry = {}
			entry['time'] = data[i][0]
			m = 0# np.max()
			for j in range(4):
				for k in range(50):
					val = abs(data[i+j][1][k])
					if(val > m):
						m = val


			con = None

			if twoD:
				con = list(np.asarray(data[i][1],dtype=np.float16)/m),list(np.asarray(data[i+1][1],dtype=np.float16)/m),list(np.asarray(data[i+2][1],dtype=np.float16)/m),list(np.asarray(data[i+3][1],dtype=np.float16)/m)
			else:
				con = list(np.asarray(data[i][1],dtype=np.float16)/m)+list(np.asarray(data[i+1][1],dtype=np.float16)/m)+list(np.asarray(data[i+2][1],dtype=np.float16)/m)+list(np.asarray(data[i+3][1],dtype=np.float16)/m)
			conData = np.asarray(con,dtype=np.float16)
			entry['data'] = np.asarray(conData)
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
		n = int(len(inp)*0.8)
		m = int(len(inp)*0.9)

		trX = inp[:n]
		tvX = inp[n:m]
		teX = inp[m:]
		trY = out[:n]
		tvY = out[n:m]
		teY = out[m:]

		return trX, tvX, teX, trY, tvY, teY


	if(timed):
		w,e = concatanateChannels()
		f = formatCut()
		return w,e,f
	else:
		return getTrainingTest()



if __name__=="__main__":
	# trX, tvX, teX, trY, tvY, teY = formatData(9,BASENAME,twoD=True)

	# print(trX.shape)
	# print(trX[1])
	# print(trX[2])
	# plt.plot(trX[4][1])
	# plt.show()

	for i in range(11,12):
		header, data = readfile(BASENAME+"."+str(i),[('ts','>i'),('waveform','50b')])
		print(header,len(data))
	# print(trX.shape, teX.shape, trY.shape, teY.shape)
