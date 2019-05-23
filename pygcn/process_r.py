import pickle
import numpy as np
import torch

def descret(output,thresh,n_class):
	P = []
	for j in range(len(output)):
		t = []
		for i in range(n_class):
			if output[j][i]>thresh:
				t.append(1.0)
			else:
				t.append(0.0)
	
		P.append(np.array(t))	

	return P



def ovral_Precc_Recal(p,y,n_class):



	tp,fp,fn,tn = 0,0,0,0
	for j in range(len(y)):

		for i in range(n_class):
			ind = p[j][i]
			if ind == 1.0 and y[j][i]==1.0:
				tp+=1
			if ind == 1.0 and y[j][i]==0.0:
				fp+=1
			if ind == 0.0 and y[j][i]==1.0:
				fn+=1
			if ind == 0.0 and y[j][i]==0.0:
				tn+=1



	print ('TP',tp)
	print ('FP',fp)
	print ('FN',fn)
	print ('TN',tn)

	if tp == 0:
		pre = 0
		rec = 0

	else:
		pre = float(tp)/float(fp+tp)
		rec = float(tp)/float(tp+fn)

	print ('Precision: ', pre)
	print ('Recal: ', rec)

def unseen_ovral_Precc_Recal(p,y,unseen,n_class):

	tp,fp,fn,tn = 0,0,0,0
	for j in range(len(y)):

		for i in range(n_class):
			if i not in unseen:
				ind  = p[j][i]
				if ind == 1.0 and y[j][i]==1.0:
					tp+=1
				if ind == 1.0 and y[j][i]==0.0:
					fp+=1
				if ind == 0.0 and y[j][i]==1.0:
					fn+=1
				if ind == 0.0 and y[j][i]==0.0:
					tn+=1



	print ('TP',tp)
	print ('FP',fp)
	print ('FN',fn)
	print ('TN',tn)
	
	if tp == 0:
		pre = 0
		rec = 0

	else:
		pre = float(tp)/float(fp+tp)
		rec = float(tp)/float(tp+fn)

	print ('Precision: ', pre)
	print ('Recal: ', rec)




def prec_recal(P, Y):
	n_class = 128

	P = np.array(P)
	Y = np.array(Y)
	
	c_P = P.T
	c_Y = Y.T

	
	
	c_TrP, c_FalP, c_FalN = [0 for i in range(n_class)],[0 for i in range(n_class)],[0 for i in range(n_class)]
	for i in range(n_class):
		X = c_P[i] - c_Y[i]
		
		for j in range(len(X)):
			if X[j] == 0.0 and c_P[i,j] == 1.0:
				c_TrP[i]+=1.0
			if X[j] == 1.0:
				c_FalP[i]+=1.0
			if X[j] == -1.0:
				c_FalN[i]+=1.0
	
	c_Prec = []
	c_Recal = []
	for i in range(n_class):
		if c_TrP[i] == 0:
			c_Prec.append(0)
			c_Recal.append(0)
		else:
			
			c_Prec.append(float(c_TrP[i])/float(c_FalP[i]+c_TrP[i]))
			c_Recal.append(float(c_TrP[i])/float(c_FalN[i]+c_TrP[i]))
	
	return c_Prec, c_Recal
	
def get_mAP(Pr, Re):
	
	n_class = 128
	Pr = np.array(Pr).T
	Re = np.array(Re).T
	
	c_mAP = [0 for i in range(n_class)]
	
	for i in range(n_class):
		f = 1.49
		nbin = int(np.rint(f)+1)

		bn = [[0] for _ in range(nbin)]

		for j in range(Pr[i].size):
			bn[int(np.rint((f)*Re[i,j]))].append(Pr[i,j])

		c_mAP[i] = np.mean([np.max(bn[_]) for _ in range(nbin)])

	return c_mAP


def classwise_mAP(y,p,unseen,classes_n):
	
		

	batch_size = 50

	n_iter = len(y)/batch_size

	f=0

	Prec, Recal = [], []
	for i in range(n_iter):
		t = f+batch_size 

		c_Prec, c_Recal = prec_recal(p[f:t],y[f:t])
	
		Prec.append(c_Prec)
		Recal.append(c_Recal)

		f = t



	c_mAP = get_mAP(Prec, Recal)

	s = 0
	k = 0
	for i in range(128):
		if i not in unseen:
			k+=1
			s += c_mAP[i]
			print (c_mAP[i],classes_n[i])

	s = float(s)/float(k)

	print ('Average mAP: ', s)






with open('128/pred_151_120_SL1_unseen.pkl') as f:
	output = pickle.load(f)

output = [i.detach().cpu().numpy() for i in output]

with open('128/y_128_full.pkl') as f:
	y = pickle.load(f)

unseen = []
with open('128/unseen_128.txt') as f:
	while True:
		s = f.readline()
		if s == '':
			break
		s = s.strip().split()[0]
		unseen.append(int(s))

with open('128/classes_128.pkl') as f:
	classes = list(pickle.load(f))

y = y[60000:]

for thresh in [0.03,0.031,0.032,0.033,0.034,0.035,0.036,0.037,0.038,0.039]:
	print (thresh)
	p = descret(output,thresh,len(classes))
	ovral_Precc_Recal(p,y,len(classes))
	

























