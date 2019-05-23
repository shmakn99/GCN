from __future__ import division
from __future__ import print_function

import time
import pickle
import argparse
import random as rd
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd

from torch import nn

from utils import load_data, accuracy
from models import GCN


with open('128/glove_128.pkl') as f:
	gl = pickle.load(f)

def embed_x(x):


	x_u = np.zeros((128,300))
	for i in range(128):
		x_u[i] = x[i]*gl[i]

	return torch.Tensor(x_u)

def embed_y(y):

	y_u = []
	for i in range(128):
		if y[i] == 1.0:
			y_u.append(np.array([0,1]))
		else:
			y_u.append(np.array([1,0]))	
	
	y_u = np.array(y_u)

	return torch.Tensor(y_u)


def load_data():
	print ('loading data')
	
	with open('31_365/y_31_365_full.pkl') as f:
		yl = pickle.load(f)

	with open('31_365/x_31_365_full.pkl') as f:
		xl = pickle.load(f)

	with open('365/adj_365_2D_renorm.pkl') as f:
		adjl = pickle.load(f)

	return yl,xl,adjl

torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

print ('loading data')
x,y,adj = load_data()



y_train = y[:15000]
x_train = x[:15000]
y_test = y[15000:]
x_test = x[15000:]
adj = torch.Tensor(adj)


print ('loading on cuda')

model = GCN(nfeat=1,
            nhid=5,
            nclass=1,
            dropout=0.5)

optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)


model.cuda()

adj = adj.cuda()



def train(epoch):
#	LR = [1.00e-02, 8.89e-03, 7.78e-03, 6.67e-03, 5.56e-03, 4.45e-03, 3.34e-03, 2.23e-03, 1.12e-03, 1.00e-05]
	print ('Epoch: '+str(epoch))

	n_iter = 500

	batch_size = int(15000/n_iter)

	t = time.time()
	model.train()

	Loss = nn.SmoothL1Loss()

#	for g in optimizer.param_groups:
#   		g['lr'] = LR[epoch]
	
	for itn in range(n_iter):

		optimizer.zero_grad()
		loss_train = 0
		
		batch = rd.sample([i for i in range(15000)],batch_size)
		
		for _ in batch:
#			x_s = torch.stack([x[_][None,:].permute(1,0) for i in range(300)])
		
			x_s = torch.Tensor(x_train[_]).float()[None, :].permute(1,0).cuda()
			y_s = torch.Tensor(y_train[_]).float().cuda()
			
		
#			print(x_s.size())
			output = model(x_s, adj)
			
#			print (output.size(), y_s.size())
	
			
			loss_train += Loss(output, y_s)

		
		loss_train/=batch_size

		if itn%100 == 0:
			print ('Iter: '+str(itn)+ ' Sum Measure: '+str(torch.sum(output))+' '+str(torch.sum(y_s)))
			
		if itn%100 == 0:
			print ('Iter: '+str(itn)+ ' Loss: '+str(loss_train))

		

		loss_train.backward()
		optimizer.step()


	print('Epoch: {:04d}'.format(epoch+1),
          'time: {:.4f}s'.format(time.time() - t))

def test():
	print ('testing')
	model.eval()
	O=[]
	for _ in range(len(x_test)):
		x_s = torch.Tensor(x_train[_]).float()[None, :].permute(1,0).cuda()
		output = model(x_s, adj)

		O.append(output)

	with open('31_365/pred_1_5_1_30_SL1.pkl','w') as f:
		pickle.dump(O,f)

		
	

t_total = time.time()
for epoch in range(10):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


test()

print ('saving')
# save model name  - GCN_nfeat_nhid1..._nclass_batchsize
torch.save(model.state_dict(), '31_365/models/GCN_1_5_1_30_SL1.pt')
