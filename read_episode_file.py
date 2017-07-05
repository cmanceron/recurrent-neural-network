import numpy as np
import matplotlib.pyplot as plt
import math
"""
	Read the joints file and return an array with the numbre of the known frames and also an array with the coordinate of the points
"""
def episode_recovery_data(name_of_the_file):
	file = open(name_of_the_file, "r")
	data = list()
	for line in file:
		data.append(line)
	file.close()

	# How many known frames ?
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

def absolute_to_relative_data(Y_frames):

	# Make the data relative to the head 
	nb_of_data,nb_of_mesurements = Y_frames.shape
	relative_Y_frames = np.zeros_like(Y_frames)
	for i in xrange(nb_of_data):
		# Head coordinates and distance between the head and the sensor
		referentiel_x = Y_frames[i][9]
		referentiel_y = Y_frames[i][10]
		referentiel_z = Y_frames[i][11]
		distance = math.sqrt((referentiel_x*referentiel_x) +(referentiel_y*referentiel_y) + (referentiel_z*referentiel_z))
		
		# Make X,Y,Z relative to the head 
		for j in xrange (nb_of_mesurements):
			if j%3 == 0:
				relative_Y_frames[i][j]=(Y_frames[i][j]-referentiel_x)/distance
			elif j%3 == 1:
				relative_Y_frames[i][j]=(Y_frames[i][j]-referentiel_y)/distance
			elif j%3 == 2:
				relative_Y_frames[i][j]=(Y_frames[i][j]-referentiel_z)/distance
	relative_Y_frames =  np.delete(relative_Y_frames,[9,10,11],1)
	#print relative_Y_frames[:][9]
	return relative_Y_frames
			


"""
	Interpolate the data with the known frames and a step
"""
def data_interpolation(X_frames,Y_frames,step):
	scope = X_frames[len(X_frames)-1]
	nb_of_known_frames,nb_of_mesurements = Y_frames.shape
	y=np.empty(nb_of_known_frames)
	interpolated_Y = np.zeros((int((scope)/step +2 ),nb_of_mesurements))
	interpolated_X = np.zeros((int((scope)/step +2)))
	for i in xrange(nb_of_mesurements):
		k=0
		for j in xrange(0,int(scope+step),int(step)):
			if i ==0:
				interpolated_X[k]=j
			
			interpolated_Y[k][i]=np.interp(j,X_frames,Y_frames[:,i])
			k +=1
	"""
	#Show the interpolation for one data
	plt.figure(1)
	plt.plot(X_frames, Y_frames[:,0], 'bo',interpolated_X ,interpolated_Y[:,0], 'k')
	plt.show()
	"""
	return interpolated_X, interpolated_Y

#X_frames,Y_frames=episode_recovery_data("joints_s10_e02.txt")
#relative_Y_frames=absolute_to_relative_data(Y_frames)
#interpolated_X, interpolated_Y=data_interpolation(X_frames,relative_Y_frames,5)
