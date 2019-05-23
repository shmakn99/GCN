import pickle
import numpy as np
import networkx as nx
import random as rd




def create_unseen_list(N):
	classwise_numbers = []
	with open('31/classwise_number_31.txt') as f:
		while True:
			s = f.readline()
			if s == '':
				break
		
			s = "".join(_ for _ in s if _ in "1234567890")

			classwise_numbers.append(int(s))
	with open('31/classes_31.pkl') as f:
		classes = list(pickle.load(f))


	s_classwise_numbers = sorted(classwise_numbers)
	#print(s_classwise_numbers)
	ind = (np.argsort(classwise_numbers))

	unseen = []

	x = [[] for i in range(1000)]
	for i in range(len(s_classwise_numbers)):
		x[(s_classwise_numbers[i]/100)].append(ind[i])

	x = [i for i in x if len(i)>1]

	unseen = rd.sample([rd.sample(i,1)[0] for i in x],N)

	with open('31/unseen_31.txt','w') as f:
		for i in unseen:
			f.write(str(i)+' '+classes[i]+'\n')

	


def create_unseen_x():
	with open('31/y_31_full.pkl') as f:
		y = pickle.load(f)

	x = []

	unseen = []
	with open('31/unseen_31.txt') as f:
		while True:
			s = f.readline()
			if s == '':
				break
			s = s.strip().split()[0]
			unseen.append(int(s))

	print (unseen)

	for i in range(len(y)):
		t = []
		for j in range(len(y[i])):
			if j in unseen:
				t.append(0.0)
			else:
				t.append(y[i][j])
		x.append(np.array(t))

	print (len(y))
	print (len(x))
	print (type(x[345]))
	print (x[789].shape)

	with open('31/x_31_unseen.pkl','w') as f:
		pickle.dump(x,f)

create_unseen_x()
