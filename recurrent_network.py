#### Libraries
# Standard library
import random
import copy


# Third-party libraries
import numpy as np
import cPickle as pickle
import pickle

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def tanh(x):
	output=(2/(1+np.exp(-2*x)))-1
	return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def tanh_output_to_derivative(output):
	return 1-(output*output)

def activation_function(function_type,output):
	if function_type == 0:
		return sigmoid(output)
	elif function_type == 1:
		return tanh(output)

def output_to_derivative(function_type,output):
	if function_type == 0:
		return sigmoid_output_to_derivative(output)
	elif function_type == 1:
		return tanh_output_to_derivative(output)



class RecurrentNetwork(object):

	def __init__(self,number_of_input,number_of_output,number_of_hidden_nodes=10,learning_rate=0.05, function_type=0):
		self.nb_input = number_of_input
		self.nb_output = number_of_output
		self.nb_hidden_nodes = number_of_hidden_nodes
		self.clip = 1
		self.alpha = learning_rate
		self.function_type = function_type

		np.random.seed(0)
		self.synapse_0 = 2*np.random.random((self.nb_input,self.nb_hidden_nodes)) - 1
		self.synapse_1 = 2*np.random.random((self.nb_hidden_nodes,self.nb_output)) - 1
		self.synapse_h = 2*np.random.random((self.nb_hidden_nodes,self.nb_hidden_nodes)) - 1

	""" 
		Train the neuronal network
		input :
			- self
			- training_data_input et training_data_output
				list of numpy array [shape = (number of input/output, length of the sequence)
				length of the list is the number of examples
			- clip (1 -> reset the hidden_layer else keep it)
			- print_out ( True -> print the progress / False -> do not print it)
		output : none
	"""


	def training (self,training_data_input,training_data_output,clip=0,print_out=True):
		synapse_0_update = np.zeros_like(self.synapse_0)
		synapse_1_update = np.zeros_like(self.synapse_1)
		synapse_h_update = np.zeros_like(self.synapse_h)
		true_positive = 0
		false_positive = 0
		true_negative = 0
		false_negative = 0
		good_answer = 0
		overallError = 0
		temp=0
    	# training logic
		if clip == 1:
			self.synapse_h = 2*np.random.random((self.nb_hidden_nodes,self.nb_hidden_nodes)) - 1
		for j in range(len(training_data_input)):
			# Data recovery
			(nb_input_data,len_seq_data)=training_data_input[j].shape
			(nb_output_data,len_output_data)=training_data_output[j].shape		
			if (self.nb_input != nb_input_data):
				print("ERROR : Training_data (nb_input) don't match with the neural network")
			elif (self.nb_output != nb_output_data):
				print("ERROR : Training_data (nb_output) don't match with the neural network")
	    	
			############## 	Can be change	###############
			#The following part is only here to prove that the neural network work for an addition, and allows us to see how it learns
			#You may want to change it for a different application of the, anyways it will not change the neural network
			true_output = training_data_output[j]
			predicted_output = np.zeros_like(true_output)
			overallError = 0
			###############################################
			
			layer_2_deltas = list()
			layer_1_values = list()
			layer_1_values.append(np.zeros(self.nb_hidden_nodes))
	       
	        # moving along the sequence
			for position in range(len_seq_data):
	            
	            # generate input and output
				X=np.atleast_2d(training_data_input[j][:,len_seq_data-position-1])
				y=np.atleast_2d(training_data_output[j][:,len_seq_data-position-1].T)

	            # hidden layer (input ~+ prev_hidden)
				temp=np.dot(X,self.synapse_0) + np.dot(layer_1_values[-1],self.synapse_h)
				layer_1 = activation_function(self.function_type,temp)
				layer_2 = activation_function(self.function_type,(np.dot(layer_1,self.synapse_1)))

				############## 	Can be change	###############	
				if layer_2>0.5:
					predicted_output[0,len_seq_data - position - 1 ]=1
					if true_output[0,len_seq_data - position - 1 ] == 1:
						true_positive+=1
					else :
						false_positive+=1
				else:
					predicted_output[0,len_seq_data - position - 1 ]=0
					if true_output[0,len_seq_data - position - 1 ] == 0:
						true_negative +=1
					else:
						false_negative+=1
				###############################################

	            # did we miss?... if so, by how much?
				layer_2_error = y - layer_2

				layer_2_deltas.append((layer_2_error)*output_to_derivative(self.function_type,layer_2))
 
				overallError += np.abs(layer_2_error[0])
	            
	            # store hidden layer so we can use it in the next timestep
				layer_1_values.append(copy.deepcopy(layer_1))
			
			
			if str(predicted_output)==str(true_output):
				good_answer+=1

			future_layer_1_delta = np.zeros(self.nb_hidden_nodes)
	        
			for position in range(len_seq_data):
	            
				X = np.atleast_2d(training_data_input[j][:,position])
				layer_1 = layer_1_values[-position-1]
				prev_layer_1 = layer_1_values[-position-2]
	            
	            # error at output layer
				layer_2_delta = layer_2_deltas[-position-1]
	            # error at hidden layer
				layer_1_delta = (future_layer_1_delta.dot(self.synapse_h.T) + layer_2_delta.dot(self.synapse_1.T)) * output_to_derivative(self.function_type,layer_1)
				#print layer_2_delta
	            # let's update all our weights so we can try again
				synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
				synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
				synapse_0_update += X.T.dot(layer_1_delta)
	            
				future_layer_1_delta = layer_1_delta
			
			self.synapse_0 += synapse_0_update * self.alpha
			self.synapse_1 += synapse_1_update * self.alpha
			self.synapse_h += synapse_h_update * self.alpha   

			synapse_0_update *= 0
			synapse_1_update *= 0
			synapse_h_update *= 0

			
		############## 	Can be change	###############
		if print_out == True:
			print ("\n*********************** Result ***********************")
			print ("Good Answers/Total = "+str(good_answer))+" / "+str(len(training_data_input))
			if good_answer!=len(training_data_input):
				print ("-------------------------------")
				print ("---> false_negative = "+str(false_negative))
				print ("---> false_positive = "+str(false_positive))
				print ("-------------------------------")
				print ("---> true_negative = "+str(true_negative))
				print ("---> true_positive = "+str(true_positive))
			print ("******************************************************\n")
		###########################################

		return overallError


	def testing(self,testing_data_input,testing_data_output,print_out = True,clip=0):
		good_answer=0
		true_positive = 0
		false_positive = 0
		true_negative = 0
		false_negative = 0
		good_answer = 0
		nb_tests=len(testing_data_input)
		synapse_h_update = np.zeros_like(self.synapse_h)
		synapse_1_update = np.zeros_like(self.synapse_1)
    	# testing logic
		if clip == 1:
			self.synapse_h = 2*np.random.random((self.nb_hidden_nodes,self.nb_hidden_nodes)) - 1
		for j in range(nb_tests):
			# Data recovery
			(nb_input_data,len_seq_data)=testing_data_input[j].shape
			(nb_output_data,len_output_data)=testing_data_output[j].shape		
			if (self.nb_input != nb_input_data):
				print("ERROR : Testing_data (nb_input) don't match with the neural network")
			elif (self.nb_output != nb_output_data):
				print("ERROR : Testing_data (nb_output) don't match with the neural network")


			############## 	Can be change	###############
			#The following part is only here to prove that the neural network work for an addition, and allows us to see how it learns
	    	#You may want to change it, anyways it will not change the neural network
			true_output = testing_data_output[j]
			predicted_output = np.zeros_like(true_output)
			overallError = 0
			###############################################

			layer_2_deltas = list()
			layer_1_values = list()
			layer_1_values.append(np.zeros(self.nb_hidden_nodes))
	        
	        # moving along the positions in the binary encoding
			for position in range(len_seq_data):
	            
				# generate input and output
				X=np.atleast_2d(testing_data_input[j][:,len_seq_data-position-1])
				y = np.atleast_2d(testing_data_output[j][:,len_seq_data-position-1].T)

				# hidden layer (input ~+ prev_hidden)
				layer_1 = activation_function(self.function_type,(np.dot(X,self.synapse_0) + np.dot(layer_1_values[-1],self.synapse_h)))

				# output layer (new binary representation)
				layer_2 = activation_function(self.function_type,(np.dot(layer_1,self.synapse_1)))

				# did we miss?... if so, by how much?
				layer_2_error = y - layer_2
				layer_2_deltas.append((layer_2_error)*output_to_derivative(self.function_type,layer_2))
				overallError += np.abs(layer_2_error[0])

	            # store hidden layer so we can use it in the next timestep
				layer_1_values.append(copy.deepcopy(layer_1))
				
				############## 	Can be change	###############	
				if layer_2>0.5:
					predicted_output[0,len_seq_data - position - 1 ]=1
					if true_output[0,len_seq_data - position - 1 ] == 1:
						true_positive+=1
					else :
						false_positive+=1
				else:
					predicted_output[0,len_seq_data - position - 1 ]=0
					if true_output[0,len_seq_data - position - 1 ] == 0:
						true_negative +=1
					else:
						false_negative+=1
			if str(predicted_output)==str(true_output):
				good_answer+=1

		print ("\n*********************** Result ***********************")
		print ("Good Answers/Total = "+str(good_answer))+" / "+str(len(testing_data_input))
		if good_answer!=len(testing_data_input):
			print ("-------------------------------")
			print ("---> false_negative = "+str(false_negative))
			print ("---> false_positive = "+str(false_positive))
			print ("-------------------------------")
			print ("---> true_negative = "+str(true_negative))
			print ("---> true_positive = "+str(true_positive))
		print ("******************************************************\n")
		###############################################


	def save(self, filename):
		 pickle.dump(self, open( filename, "wb" ) )
	
	def load(self,filename):
		up= pickle.load( open( filename, "rb" ) )
		return up
	def save_in_a_file(self,name_of_the_file):
		file=open(name_of_the_file,"w")
		file.write("Network\n")
		file.write("Number of inputs = " +str(self.nb_inputs) +"\n")
		file.write("Number of outputs = " +str(self.nb_outputs)+"\n") 
		file.write("Number of hidden nodes = " +str(self.nb_hidden_nodes)+"\n") 
		file.write("------------------------------------\n")
		file.write("*** Synapse_0 ***\n")
		file.write(str(self.synapse_0))
		file.write("------------------------------------\n")
		file.write("*** Synapse_1 ***\n")
		file.write(str(self.synapse_1))
		file.write("------------------------------------\n")
		file.write("*** Synapse_h ***\n")
		file.write(str(self.synapse_h))
		file.write("------------------------------------\n")