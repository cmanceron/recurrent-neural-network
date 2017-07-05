import numpy as np

from read_actionLabel_file import actionLabel_recovery_data
from read_episode_file import episode_recovery_data,absolute_to_relative_data,data_interpolation

"""
	Assign 1 to the frames if it is the behaviour wanted, 0 if not
"""
def output_assignement(what_you_want_to_be_at_one,episode_label,size_of_the_scope,step):

	#Recovery of the behaviour's frames
	output_equal_1 = episode_label[what_you_want_to_be_at_one]
	
	#Round to the next step
	if output_equal_1[0]%step !=0:
		output_equal_1[0] = int(step * round(output_equal_1[0]/step)+5)
	if output_equal_1[1]%step !=0:
		output_equal_1[1] = int(step * round(output_equal_1[1]/step)+5)
	
	#Initiate to 0 the output
	output_values = np.zeros(size_of_the_scope)
	
	# Change to one if it is the behaviour wanted
	for i in xrange (len(output_values)):
		if i >= output_equal_1[0]/step and i <=output_equal_1[1]/step:
			output_values[i] = 1
	return output_values



def get_input_and_output(name_of_actionLabel,step,what_you_want_to_be_at_one):
	actionLabel=actionLabel_recovery_data(name_of_actionLabel)
	input_values = list()
	output_values = list()
	for i in xrange(len(actionLabel)):
		current_file_name = str("joints_"+actionLabel[i]['name']+".txt")
		current_X_frames,current_Y_frames = episode_recovery_data(current_file_name)
		current_relative_Y_frames = absolute_to_relative_data(current_Y_frames)
		current_interpolated_X, current_interpolated_Y = data_interpolation(current_X_frames,current_relative_Y_frames,step)
		current_size = len(current_interpolated_X)
		current_output_values = output_assignement(what_you_want_to_be_at_one,actionLabel[i],current_size,step)
		output_values.append(current_output_values)
		input_values.append(current_interpolated_Y)
	return input_values,output_values


