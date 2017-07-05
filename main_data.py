from output_assignment import get_input_and_output
import recurrent_network as rnw
import numpy as np
import pickle
#def convert_into_good_shape(input_values,output_values)


"""
	Convert the data to shape that the neural network can process
"""
def convert_data (input_values,output_values):
	data_input=list()
	data_output=list()
	for i in xrange(len(input_values)):
		(nb_of_frames,nb_of_mesurement)=input_values[i].shape
		data_input.append(input_values[i].T)
		data_output.append(np.atleast_2d(output_values[i].T))
	return data_input,data_output
"""
	Distribute the data between a training set and a test set
"""
def training_test_distribution (stake_of_training,data_input,data_output):
	number_of_training = int(len(data_output)*stake_of_training)
	number_of_test = len(data_output)- number_of_training

	training_data_input = list()
	training_data_output = list()
	testing_data_input = list()
	testing_data_output = list()

	training_data_input = data_input[0:number_of_training-1]
	testing_data_input = data_input[number_of_training:-1]

	training_data_output = data_output[0:number_of_training-1]
	testing_data_output = data_output[number_of_training:-1]

	return training_data_input,training_data_output,testing_data_input,testing_data_output


net = rnw.RecurrentNetwork(57,1,number_of_hidden_nodes = 50,learning_rate =0.01,function_type=0)
#net = pickle.load(open("save.p","rb"))
input_values,output_values=get_input_and_output("actionLabel.txt",5,"walk")
data_input,data_output=convert_data (input_values,output_values)
training_data_input,training_data_output,testing_data_input,testing_data_output  = training_test_distribution(0.8,data_input,data_output)
saving_the_weights = list()
current_weights = list()
overallError = np.zeros([6001])
print("******* Training *******")
for i in xrange (6000):

	print_boolean = False
	if i%10 ==0:
		current_weights=[]
		current_weights.append(net.synapse_0)
		current_weights.append(net.synapse_1)
		current_weights.append(net.synapse_h)
		saving_the_weights.append(current_weights)
		if i%100==0 :
			print_boolean = True
			net.save("save.p")
			pickle.dump(saving_the_weights,open("saving_the_weights.p", "wb"))
			pickle.dump(overallError,open("saving_overallError.p","wb"))
	overallError[i]= net.training(training_data_input,training_data_output,print_out=print_boolean)
	print str(i)+ "  " +str(overallError[i])
net.save("save.p")
pickle.dump(saving_the_weights,open("saving_the_weights.p", "wb"))
pickle.dump(overallError,open("saving_overallError.p","wb"))

#print("******* Testing *******")
#net.testing(testing_data_input,testing_data_output)



#pb derniere valeur input values
