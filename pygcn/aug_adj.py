import pickle
import numpy as np



def renorm(adj):
	Dt = np.zeros((81,81))


	for i in range(81):
		adj[i,i] = adj[i,i]+1.0
	

	for i in range(81):
	
		Dt[i,i] = -float(1)/float(np.sqrt(np.sum(adj[i])))

	augA = np.matmul(np.matmul(Dt,adj),Dt)

	return augA

with open('coco/KG_coco.pkl') as f:
	G = pickle.load(f)


classes = []
with open('coco/classes_coco.txt') as f:
	while True:
        	s = f.readline().strip('\n')
		if s == '':
			break
		classes.append(s)


adj = np.zeros((81,81))
adjw = np.zeros((81,81))

ndlist = G.nodes()


for ed in G.edges():
        if ed[0] in classes and ed[1] in classes:
		print (ed[0], ed[1])
	        adj[(classes.index(ed[0]), classes.index(ed[1]))] = G[ed[0]][ed[1]]['weight']



adjr = renorm(adjw)

with open('coco/adj_coco_2D.txt','w') as f:
	for i in range(81):
		f.write(''.join(str(adj[i].tolist()))+'\n')

with open('coco/adj_coco_2D_weighted.txt','w') as f:
	for i in range(81):
		f.write(''.join(str(adjw[i].tolist()))+'\n')


with open('coco/adj_coco_2D_renorm.txt','w') as f:
	for i in range(81):
		f.write(''.join(str(adjr[i].tolist()))+'\n')

