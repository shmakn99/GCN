import json
import numpy as np
import pickle

def xyz(G):

	objects =  list(G.nodes())

	with open('relationships.json') as f:
		relationships = json.load(f)


	i=0 


	images = []



	for img in relationships:
		if i%1000 == 0:
			print (i)
		i+=1

		for rlshn in img['relationships']:
			if 'name' in rlshn['object'].keys():
				name  = rlshn['object']['name']
			else:
				name = rlshn['object']['names'][0] 
			if name in objects:
				images.append(img['image_id'])		
		

	images = list(set(images))
	print (len(set(images)))

	with open('coco/images_coco.txt','w') as f:
		for i in images:
			f.write(str(i)+'.jpg'+'\n')


with open('coco/KG_coco.pkl') as f:
	G = pickle.load(f)

xyz(G)

classes = list(G.nodes())

with open('coco/classes_coco.txt','w') as f:
	for c in classes:
		f.write(c+'\n')
