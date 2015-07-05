from reading_and_viewing_data.readDACQfile import _readFile as readfile 
import sys
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "../R2192/20140110_R2192_track1.pos"

def getXY(filename=FILENAME):
	header, data = readfile(filename,[('ts','>i'),('pos','>8h')])
	x = [x for x,_,_,_,_,_,_,_ in data['pos']]
	y = [y for _,y,_,_,_,_,_,_ in data['pos']]
	for n,i in enumerate(x):
		if i>800:
			x[n] = 640
	for n,i in enumerate(y):
		if i>800:
			y[n] = 500

	x = np.asarray(x,dtype=np.float16)
	y = np.asarray(y,dtype=np.float16)

	xmin = np.amin(x)
	xmax = np.amax(x)
	ymin = np.amin(y)
	ymax = np.amax(y)

	# print("X min",xmin)
	# print("X max",xmax)
	# print("Y min",ymin)
	# print("Y max",ymax)

	diffx = xmax - xmin
	diffy = ymax - ymin

	scale = np.amax([diffx,diffy])

	# print("Scale ", scale)

	for n,i in enumerate(range(len(x))):
		x[i] = x[i] - xmin
		x[i] = x[i]* (1.0/scale)


	for n,i in enumerate(range(len(y))):
		y[i] = y[i] - ymin
		y[i] = y[i]*1.0/scale

	return x, y

if __name__=="__main__":
	x, y = getXY()


	# print(data['pos'][1000:1030])
	# plt.plot(data['pos'])
	# plt.show()
	plt.scatter(x, y)
	plt.show()
	# with open('pos.txt','w') as f:
	# 	f.write(str(x) + " " + str(y))