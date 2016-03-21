"""
# Makes a song shorter by analyzing similarities to find shortcuts
# mostly trying to recreate this : 
# http://labs.echonest.com/Uploader/index.html?trid=TRWQFRH1430BC48B32
#
# Copyright (c) 2015 Antoine Letouzey antoine.letouzey@gmail.com
# Author: Antoine Letouzey
# LICENSE: LGPL
"""

from pylab import plot, show, title, xlabel, ylabel, subplot, savefig, axvline, pcolor, figure
import matplotlib.pyplot as plt
import pylab
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
import numpy as np
from scipy.io.wavfile import read,write
from graph import Graph
import os

if not os.path.exists('./samples'):
    os.makedirs('./samples')


Fs = 44100.0  # sampling rate wav
#FILE, BPM, sample_path = 'crazy.wav', 112.0, 'samples/crazy_%03d.wav' # crazy
FILE, BPM, sample_path = 'maybe.wav', 120.0, 'samples/maybe_%03d.wav' # call me maybe
size = int(Fs*60/BPM)
NBSHORTCUT = 30

rate,data=read(FILE)
y=data[:,0]
lungime=len(y)
timp=len(y)/Fs
#print timp
t=linspace(0,timp,len(y))
#figure()
#plot(t,y)
#i = 122
#plot(range(size), y[i*size:(i+1)*size])
#i = 377
#plot(range(size), y[i*size:(i+1)*size])

#for i in np.arange(0,timp,60/BPM):
#    axvline(i, color='r', linestyle='solid')
#xlabel('Time')
#ylabel('Amplitude')


#correlation matrix

 # number of samples per beat
nbsamples = int(len(y)/size)
#print nbsamples

#save samples if needed
if not os.path.exists(sample_path%1):
	print "saving samples ... ",
	for i in xrange(nbsamples):
		write(sample_path%i, 44100, y[i*size:(i+1)*size])
	print "Done."

corrMat = np.zeros((nbsamples, nbsamples));

for i in xrange(nbsamples-1):
    s1 = np.array(abs(y[i*size:(i+1)*size])+1, dtype=np.float32)
    s1 /= np.sum(s1)
    for j in xrange(i, nbsamples):
        s2 = np.array(abs(y[j*size:(j+1)*size])+1, dtype=np.float32)
        s2 /= np.sum(s2)
        corrMat[j,i] = np.sum(np.abs(s1-s2))/float(size)#np.correlate(s1, s2)/float(size)
orriCorMat = corrMat.copy()



cmax = np.max(corrMat)

# fill begining, end and diagonal
corrMat[(nbsamples-20):nbsamples,:] = cmax
corrMat[:, 0:10] = cmax
corrMat[corrMat == 0] = cmax

cmin = np.min(corrMat)

#print cmin, cmax
figure()
pcolor(corrMat)
pylab.colorbar()
pylab.axis('equal')


# circle
#polar coordiantes
pos = np.array([ [np.cos(a), np.sin(a)] for a in [-((float(i)/nbsamples) * 2 * np.pi - np.pi/2.0)  for i in xrange(nbsamples)]])

figure()
plot(pos[:,0], pos[:,1])
pylab.axis('equal')

pad = 1

print "shortcuts"
adj = np.zeros((nbsamples, nbsamples))
# link every sample to its following one
for i in xrange(nbsamples-1) : adj[i, i+1] = 1
SC = []
for i in xrange(NBSHORTCUT) :
    p = np.where(corrMat == np.min(corrMat))
    print [ int(e) for e in p[::-1]]
    SC.append(list(p[::-1]))
    adj[p[0], p[1]] = 1
    adj[p[1], p[0]] = 1
    plot([pos[p[1],0], pos[p[0],0]], [pos[p[1],1], pos[p[0],1]])
    corrMat[p[0]-pad:p[0]+pad, p[1]-pad:p[1]+pad] = cmax
    s1 = y[p[0]*size:(p[0]+1)*size]
    s2 = y[p[1]*size:(p[1]+1)*size]

figure()
for i in xrange(9):
	plt.subplot(331+i)
	plt.plot(range(size), y[SC[i][0]*size:(SC[i][0]+1)*size], 'b')
	plt.plot(range(size), y[SC[i][1]*size:(SC[i][1]+1)*size], 'g')
	plt.title('%d - %d : %f'%(SC[i][0], SC[i][1], orriCorMat[SC[i][1], SC[i][0]])) 



print "\nPATH"
graph = Graph(adj)
command = "sox "
prev = -1
for i, n in enumerate(graph.path(0, nbsamples-1)):
	if n != prev+1:
		print i, n, 'jump'
	else : # do not add the sample if we jumped because it's the "same"
		print i, n
		command += sample_path%n + " "
	prev = n
command += "short.wav"

print "\nHere is the full command you should type in your terminal to get the shortened version of the song :"
print command




# to keep the color the same between the two plots
corrMat[0,0] = cmin

figure()
#print np.min(corrMat), np.max(corrMat)
pcolor(corrMat)
pylab.colorbar()
pylab.axis('equal')



show()
