import matplotlib
matplotlib.use('Agg')
from reading_and_viewing_data.readDACQfile import _readFile as readfile 
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import numpy as np

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


FILENAME = "../R2192-screening/20141001_R2192_screening.pos"
# FILENAME = "../R2192/20140110_R2192_track1.pos"

VIDEO_NAME = "mouse_moving.gif"

def getXY(filename=FILENAME,sequenceLength=None,num_skip=40):
	header, data = readfile(filename,[('ts','>i'),('pos','>8h')])
	x = [x for x,_,_,_,_,_,_,_ in data['pos']]
	y = [y for _,y,_,_,_,_,_,_ in data['pos']]
	num_wrong = 0
	for n,i in enumerate(x):
		if i>1000:
			num_wrong+=1
			x[n] = x[n-1]
	for n,i in enumerate(y):
		if i>1000:
			num_wrong+=1
			y[n] = y[n-1]

	print("Num wrong: {}".format(num_wrong))

	x = np.asarray(x,dtype=np.float32)
	y = np.asarray(y,dtype=np.float32)

	xmin = np.amin(x)
	xmax = np.amax(x)
	ymin = np.amin(y)
	ymax = np.amax(y)
	# print xmin,xmax,ymin,ymax
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
		x[i] = x[i]*(1.0/scale)


	for n,i in enumerate(range(len(y))):
		y[i] = y[i] - ymin
		y[i] = y[i]*1.0/scale

	# for i in range(len(x)):
	# 	x[i] = x[i]*2.0
	# 	x[i] = x[i] - 1


	# for i in range(len(y)):
	# 	y[i] = y[i]*2.0
	# 	y[i] = y[i] - 1
	if sequenceLength == None:
		return x, y
	Z = np.asarray([x,y])
	Z = Z.transpose()
	X = []
	i = 0
	print(Z.shape)
	while(i + sequenceLength < Z.shape[0]):
		X.append(Z[i:i+sequenceLength])
		i+=num_skip
	X = np.asarray(X)
	print(X.shape)

	m = int(len(X)*0.8)
	n = int(len(X)*0.9)

	y_train = np.asarray(X[:m])
	y_valid = np.asarray(X[m:n])
	y_test = np.asarray(X[n:])

	return y_train, y_valid, y_test


def makeVideo(X, Y):

	duration = X.shape[0] / 50
	print("Duration: ",duration)
	duration = 1000
	fps = 1
	fig, ax = plt.subplots(1, figsize=(4, 4), facecolor='white')
	fig.subplots_adjust(left=0, right=1, bottom=0)
	
	def make_frame(t):
	    ax.clear()
	    ax.set_title("Rat location", fontsize=16)

	    ax.axis((0.0,1.0,0.0,1.0))
	    ax.scatter(X[50*t],Y[50*t])

	    return mplfig_to_npimage(fig)

	
	animation = mpy.VideoClip(make_frame, duration = duration)
	print(animation)
	animation.write_gif(VIDEO_NAME, fps=fps)

if __name__=="__main__":
	x, y = getXY()
	print(x.shape)
	# clip = makeVideo(x,y)
	# myclip = mpy.VideoFileClip(VIDEO_NAME)
	# print(myclip.fps)
	# myclip.preview()
	# print(data['pos'][1000:1030])
	# plt.plot(data['pos'])
	# plt.show()
	plt.scatter(x, y, lw=0.0, alpha=0.2)
	# plt.x_axis("x")
	plt.axis([0,1,0,1])
	plt.title("Positions of the rat during the trial")
	# plt.show()
	plt.savefig("pos.png")
	plt.close()
	# with open('pos.txt','w') as f:
	# 	f.write(str(x) + " " + str(y))
