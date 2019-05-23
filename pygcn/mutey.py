import random as rd
import pickle
import numpy as np



def mute(y):

	to_m_frm = [i for i in range(31) if y[i]==1]
	
	y[rd.sample(to_m_frm,1)[0]] = 0

	return y	
	

with open('31/y_31_full.pkl') as f:
	y= pickle.load(f)

x = [mute(i) for i in y]

print (len(x))

with open('31/x_31_full.pkl','wb') as f:
	pickle.dump(x,f)
