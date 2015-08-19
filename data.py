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

		timebase = float(re.sub("[^0-9]", "", header['timebase']))
		print(timebase)

		duration = int(re.sub("[^0-9]", "", header['duration']))
		print(duration)
		freq = 50.0
		time_step = 1.0/freq

		timesData = []
		inputData = []
		for i in range(0,len(data),4):
			if i+3 > len(data):
				break
			entry = {}
			entry['time'] = data[i][0] / timebase
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

		# print(int(timesData[-1]['time']))

		downsampled = []
		j = 0
		print(time_step)
		down = np.zeros(timesData[-1]['data'].shape[0])
		print(down)

		for i in range(int(duration*freq)):
			while True:
				try:
					timesData[j]
				except IndexError:
					break

				if timesData[j]['time'] < i * time_step:
					down += timesData[j]['data']
					j+=1
				else:
					break
			downsampled.append(down)
			down = np.zeros(timesData[-1]['data'].shape[0])
			j+=1

		downsampled = np.asarray(downsampled)
		print(downsampled.shape)
		return np.asarray(inputData), downsampled #timesData

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
		act,time = concatanateChannels()
		f = formatCut()
		return act,time,f
	else:
		return getTrainingTest()


if __name__=="__main__":
	a, f, t = formatData(10,BASENAME,timed=True)

	# datapoint = teX[0]

	# # plt.plot(datapoint)
	# # plt.axis([0,200,-1,1])
	# # plt.grid(True)
	# # plt.show()

	# fig = plt.figure(1)
	# sub1 = fig.add_subplot(411)
	# sub2 = fig.add_subplot(412)
	# sub3 = fig.add_subplot(413)
	# sub4 = fig.add_subplot(414)

	# # add titles
	# sub1.set_title("Example of neural spike")

	# # adding x labels

	# # sub1.set_xlabel('Time')
	# # sub2.set_xlabel('Time')
	# # sub3.set_xlabel('Time')
	# # sub4.set_xlabel('Time')


	# # adding y labels

	# sub1.set_ylabel('Channel 1')
	# sub2.set_ylabel('Channel 2')
	# sub3.set_ylabel('Channel 3')
	# sub4.set_ylabel('Channel 4')

	# Plotting data

	# print(testing[0][0])
	# inp = []
	# for z in range(4):
	#     inp += list(testing[0][0][z])


	# sub1.plot(datapoint[:50])
	# sub1.grid(True)
	# sub1.axis([0,50,-0.8,1])

	# sub2.plot(datapoint[50:100])
	# sub2.grid(True)
	# sub2.axis([0,50,-0.8,1])

	# sub3.plot(datapoint[100:150])
	# sub3.grid(True)
	# sub3.axis([0,50,-0.8,1])

	# sub4.plot(datapoint[150:200])
	# sub4.grid(True)
	# sub4.axis([0,50,-0.8,1])

	# fig.tight_layout()

	# plt.show()

	# print(trX.shape)
	# print(trX[1])
	# print(trX[2])
	# plt.plot(trX[4][1])
	# plt.show()

	# for i in range(11,12):
	# 	header, data = readfile(BASENAME+"."+str(i),[('ts','>i'),('waveform','50b')])
	# 	print(header,len(data))
	# print(trX.shape, teX.shape, trY.shape, teY.shape)
