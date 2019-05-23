import pickle
import numpy as np
import networkx as nx

def class_str(y,classes_n):
	n_class = len(classes_n)
	class_p = [0 for i in range(n_class)]


	for i in range(len(y)):
		for j in range(365):
			if y[i][j] == 1.0:
				class_p[j]+=1

	for i in range(n_class):
		print (classes_n[i], class_p[i])


def weed_out(y):
	y_u = []

	y_s = []

	
	for i in range(len(y)):
		
		if i%1000== 0:
			print (i)
		if np.sum(y[i]) > 1:
			y_u.append(y[i])
			y_s.append(np.sum(y[i]))
	

	print ('Total Object Classes', 31)
	print ('Total Images:', len(y_u))
	print ('Average Number of Objects Classes in an Image', np.mean(y_s))
	print ('Maximum number of Classes present in an Image', np.max(y_s))

	return y_u



with open('31/classes_31.pkl') as f:
	c31 = list(pickle.load(f))

with open('128/classes_128.pkl') as f:
	c128 = list(pickle.load(f))

with open('365/classes_365.pkl') as f:
	c365 = list(pickle.load(f))

map_31_128 = {}
map_31_365 = {}
map_128_365 = {}

for c in c128:
	map_128_365[c128.index(c)] = c365.index(c)

with open('128_365/map_128_365.pkl','w') as f:
	pickle.dump(map_128_365,f)


		

