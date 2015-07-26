from reading_and_viewing_data.readDACQfile import _readFile as readfile 
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.use('TkAgg')

import numpy as np

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


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

	return x, y

def makeVideo(X, Y):

	duration = X.shape[0]

	duration = duration/10

	fig, ax = plt.subplots(1, figsize=(4, 4), facecolor='white')
	fig.subplots_adjust(left=0, right=1, bottom=0)
	# xx, yy = np.meshgrid(np.linspace(-2,3,500), np.linspace(-1,2,500))

	def make_frame(t):
	    ax.clear()
	    # ax.axis('off')

	    ax.set_title("SVC classification", fontsize=16)


	    # classifier = svm.SVC(gamma=2, C=1)
	    # the varying weights make the points appear one after the other
	    # weights = np.minimum(1, np.maximum(0, t**2+10-np.arange(50)))
	    # classifier.fit(X, Y, sample_weight=weights)
	    # Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
	    # Z = Z.reshape(xx.shape)
	    # ax.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8,
	    #             vmin=-2.5, vmax=2.5, levels=np.linspace(-2,2,20))
	    # ax.scatter(X[:,0], X[:,1])
	    ax.axis((0.0,1.0,0.0,1.0))
	    ax.scatter(X,Y)

	    return mplfig_to_npimage(fig)

	animation = mpy.VideoClip(make_frame, duration = duration)
	animation.write_gif("mouse_moving.gif", fps=15)

if __name__=="__main__":
	x, y = getXY()
	makeVideo(x,y)
	# print(data['pos'][1000:1030])
	# plt.plot(data['pos'])
	# plt.show()
	# plt.scatter(x, y)
	# plt.show()
	# with open('pos.txt','w') as f:
	# 	f.write(str(x) + " " + str(y))