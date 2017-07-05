import numpy as np
import matplotlib.pyplot as plt

def episode_recovery_data(name_of_the_file):
	file = open(name_of_the_file, "r")
	data = list()
	for line in file:
		data.append(line)
	file.close()

	# How many knwon frames ?
	nb_of_known_frames = len(data)

	
	nb_of_measurements=len(data[0].split("  "))-2
	# Initiate stokage data at 0 
	X_frames = np.empty(nb_of_known_frames)
	Y_frames = np.zeros((nb_of_known_frames,nb_of_measurements))

	# Fill Y_frames with the known frames
	for i in xrange(len(data)):
		split = data[i].split("  ")
		for j in xrange (len(split)):
			if j == 0:
				X_frames[i]= split[j]
			elif split[j] != '\r\n':
				Y_frames[i][j-1]=float(split[j])
	return X_frames,Y_frames

def data_interpolation(X_frames,Y_frames,step):
	print ("Data interpolation")
	scope = X_frames[len(X_frames)-1]
	nb_of_known_frames,nb_of_mesurements = Y_frames.shape
	print Y_frames.shape
	y=np.empty(nb_of_known_frames)
	interpolated_Y = np.zeros((int((scope)/step +1 ),nb_of_mesurements))
	interpolated_X = np.zeros((int((scope)/step +1)))
	print("interpolated_Y.shape = " +str(interpolated_Y.shape))
	for i in xrange(nb_of_mesurements):
		#print i 
		y=np.empty(nb_of_known_frames)
		for j in xrange(nb_of_known_frames):
			y[j]= Y_frames[j][i]
		k=0
		for j in xrange(0,int(scope+step),int(step)):
			if i ==0:
				interpolated_X[k]=j
			interpolated_Y[k][i]=np.interp(i,X_frames,y)
			k +=1
	plt.figure()
	plt.plot(X_frames, Y_frames[0], 'o')
	plt.plot(interpolated_X, interpolated_Y[:][0], '-x')
	plt.show()
	print ("fin dessin")
	return interpolated_X, interpolated_Y
