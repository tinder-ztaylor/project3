#!/usr/bin/env python
# -*- coding: utf-8 -*-

# port to python of @karpathy's matlab code for t-SNE
# visualization of Imagenet
# see http://cs.stanford.edu/people/karpathy/cnnembed/
# download imagenet_val_embed.mat, val_imgs_med.txt there

# prerequisites: imagenet 2012 validation images
# get here: www.image-net.org
# resize images to 50x50
# set your paths appropriately in val_imgs_med.txt
# see included scripts

from __future__ import division

import scipy.io as sio
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.manifold import TSNE
import csv
import cv2


np.random.seed(1)
data = pd.read_csv('../data/intern_sample_dataset.csv', delimiter=',')

ids = []
vectors = []
genders = []
with open('../vec_reps.csv', 'rb') as file:
    for line in csv.reader(file):
        curID = line[0]
        gender = int(data.loc[data["uid"] == curID]["gender"])
        ids.append(curID)
        vectors.append(line[1:])
        genders.append(gender)

ids, vectors, genders = zip(*sorted(zip(ids, vectors, genders), key=lambda t:t[0]))

colors = ['r' if g == 0 else 'b' for g in genders]

with open('../data/valid_ids.txt', 'wb') as f:
    for uid in ids:
        #print uid
        f.write(uid + '\n')

print len(vectors[0])
model = TSNE(n_components=2, perplexity=20, random_state=0)
np.set_printoptions(suppress=True)
output = model.fit_transform(np.array(vectors))
if mpl.is_interactive():
    plt.ioff()
plt.scatter(output[:,0], output[:,1], c=colors)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Gender Breakdown')
red_patch = mpatches.Patch(color='red', label='Male')
blue_patch = mpatches.Patch(color='blue', label='Female')
plt.legend(handles=[red_patch, blue_patch])
plt.show()
# load the tsne points
data = sio.loadmat('../cnnembed/imagenet_val_embed.mat')
print type(data['x'])
print type(output)

# set min point to 0 and scale
x = output - np.min(output)
x = x / np.max(x)

# create embedding image
S = 20000  # size of full embedding image
G = np.zeros((S, S, 3), dtype=np.uint8)
s = 500  # size of single image


with open('../cnnembed/val_imgs_med.txt') as f:
	for i, fs in enumerate(f):
		if np.mod(i, 100) == 0:
			print('%d/%s...\n') % (i, str.split(fs, '/')[2])

		# set location
		a = int(np.ceil(x[i, 0] * (S-s-1)+1))
		b = int(np.ceil(x[i, 1] * (S-s-1)+1))
		a = a - np.mod(a-1,s) + 1
		b = b - np.mod(b-1,s) + 1
		if G[a,b,0] != 0:
		    continue

		I = plt.imread(fs.rstrip())

		G[a:a+s, b:b+s,:] = I

G = cv2.cvtColor(G, cv2.COLOR_RGB2BGR)
print cv2.imwrite('tsne.jpg', G)	    
#plt.imshow(G)
#plt.show()


