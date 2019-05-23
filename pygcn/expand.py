import pickle
import numpy as np


with open('128/x_128_unseen.pkl') as f:
	x128 = pickle.load(f)

#with open('128/y_128_full.pkl') as f:
#	y128 = pickle.load(f)

with open('128_365/map_128_365.pkl') as f:
	map128_365 = pickle.load(f)

x_ex = []
#y_ex = []

for j in range(len(x128)):
	nx = np.zeros(365)
#	ny = np.zeros(365)
	for i in range(128):
		nx[map128_365[i]] = x128[j][i]

#	for i in range(128):
#		ny[map128_365[i]] = y128[j][i]

	x_ex.append(nx) 
#	y_ex.append(ny)


with open('128_365/x_128_365_unseen.pkl','w') as f:
	pickle.dump(x_ex,f)

#with open('128_365/y_128_365_full.pkl','w') as f:
#	pickle.dump(y_ex,f)
