from reading_and_viewing_data.readDACQfile import _readFile as readfile 
import sys
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "../R2192/20140110_R2192_track1.pos"

def getXY(filename=FILENAME):
	header, data = readfile(filename,[('ts','>i'),('pos','>8h')])
	x = [x for x,_,_,_,_,_,_,_ in data['pos']]
	y = [y for _,y,_,_,_,_,_,_ in data['pos']]
	return np.asarray(x), np.asarray(y)

if __name__=="__main__":
	header, data = readfile(filename,[('ts','>i'),('pos','>8h')])
	print(header)
	x = [x for x,_,_,_,_,_,_,_ in data['pos']]
	y = [y for _,y,_,_,_,_,_,_ in data['pos']]

	print(data['pos'][1000:1030])
	# plt.plot(data['pos'])
	# plt.show()
	# plt.scatter(x, y)
	# plt.show()
	with open('pos.txt','w') as f:
		f.write(str(x) + " " + str(y))