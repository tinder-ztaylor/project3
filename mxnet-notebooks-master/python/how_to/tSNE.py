import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import csv
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt

ids = []
vectors = []
with open('vec_reps.csv', 'rb') as file:
    for line in csv.reader(file):
        ids.append(line[0])
        vectors.append(line[1:])

ids, vectors = zip(*sorted(zip(ids, vectors), key=lambda t:t[0]))

with open('data/valid_ids.txt', 'wb') as f:
    for uid in ids:
        #print uid
        f.write(uid + '\n')

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
output = model.fit_transform(np.array(vectors))
print(output[:,0])
if mpl.is_interactive():
    plt.ioff()
plt.scatter(output[:,0], output[:,1])
#plt.show()
