import pickle
import json
import numpy as np

images = []
with open('coco/images_coco.txt') as f:
	while True:
            s = f.readline().strip('\n')
            if s == '':
                break
            images.append(s)

classes = []
with open('coco/classes_coco.txt') as f:
	while True:
            s = f.readline().strip('\n')
            if s == '':
                break
            classes.append(s)


with open('relationships.json') as f:
	relationships = json.load(f)

def get_y(classes,rlshnps):
	
	ret = np.zeros(81)
	for rlshn in rlshnps:
		if 'name' in rlshn['object'].keys(): 	
			nameo = rlshn['object']['name']
		else:
			nameo = rlshn['object']['names'][0]

		if 'name' in rlshn['subject'].keys():
			names = rlshn['subject']['name']
		else:
			names = rlshn['subject']['names'][0]

		if nameo in classes:
			ret[classes.index(nameo)] = 1

		if names in classes:
			ret[classes.index(names)] = 1

	return ret

x = 66301

i = 0

y = []
imgid = []

for img in relationships:
	if i%1000 == 0:
		print (i)
	i+=1
        
	#print (i, img['image_id'])
	
	if str(img['image_id'])+'.jpg' in images:

		y.append(get_y(classes, img['relationships']))
                imgid.append(str(img['image_id']))
		
		


print (len(y))

with open('coco/y_coco_full.txt','w') as f:
    for i in range(len(y)):
        f.write(imgid[i]+' '+''.join(str(y[i].tolist()))+'\n')
        

















