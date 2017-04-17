import pandas as pd
import numpy as np
from os.path import join
from urllib import urlretrieve

data = pd.read_csv('data/intern_sample_dataset.csv', delimiter=',')
for uid, file_url in zip(data['uid'], data['photo 1']):
    if not isinstance(file_url, float): 
        urlretrieve(file_url, join('data', 'images', str(uid) + '.jpg'))
