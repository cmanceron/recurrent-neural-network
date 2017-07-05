import numpy as np 


"""
	Read the actionLabel and return all the information in a list of dictionnary which contains the information of the episode
"""
def actionLabel_recovery_data(name_of_the_file):
	# Opend and Read the file
	file = open(name_of_the_file, "r")
	episodes = list()
	current_episode = []
	for line in file:
		current_episode.append(line)
		if line.startswith("clapHands"):
			episodes.append(current_episode)
			current_episode=[]
	file.close()

	# data Processing
	action_label = list()
	for i in xrange(len(episodes)):
		walk = np.zeros(2)
		sitDown = np.zeros(2)
		standUp = np.zeros(2)
		pickUp = np.zeros(2)
		carry = np.zeros(2)
		throw = np.zeros(2)
		push = np.zeros(2)
		pull = np.zeros(2)
		waveHands = np.zeros(2)
		clapHands = np.zeros([0,0])
		for j in xrange(11):

			type = episodes[i][j].split(" ")
			if len(type)>2:
				if type[1]=="NaN":
					type[1] = -1
				if type[2] =="NaN\n":
					type[2] = -1

			if type[0] == "walk:":
				beginning = int(type[1])
				end = int(type[2])
				walk = np.array([beginning,end])
			elif type[0] == "sitDown:":
				beginning = int(type[1])
				end = int(type[2])
				sitDown = np.array([beginning,end])
			elif type[0] == "standUp:":
				beginning = int(type[1])
				end = int(type[2])
				standUp = np.array([beginning,end])
			elif type[0] == "pickUp:":
				beginning = int(type[1])
				end = int(type[2])
				pickUp = np.array([beginning,end])	
			elif type[0] == "carry:":
				a = 2
				beginning = int(type[1])
				end = int(type[2])
				carry = np.array([beginning,end])
			elif type[0] == "throw:":
				beginning = int(type[1])
				end = int(type[2])
				throw = np.array([beginning,end])
			elif type[0] == "push:":
				beginning = int(type[1])
				end = int(type[2])
				push = np.array([beginning,end])
			elif type[0] == "pull:":
				beginning = int(type[1])
				end = int(type[2])
				pull = np.array([beginning,end])
			elif type[0] == "waveHands:":
				beginning = int(type[1])
				end = int(type[2])
				waveHands = np.array([beginning,end])
			elif type[0] == "clapHands:":
				beginning = int(type[1])
				end = int(type[2])
				clapHands = np.array([beginning,end])
			else:
				name = type[0].split('\n')[0]
		episode_dict =  {'name':name,'walk': walk,'sitDown':sitDown,'standUp':standUp, 'pickUp':pickUp,'carry':carry,'throw':throw,'push':push,'pull':pull}
		action_label.append(episode_dict)

	return action_label