#!/usr/bin/env python
# -*- coding: utf-8 -*-

# resize imagenet 2012 val images

import os
import glob
import cv2

# path to imagenet validation data (50k images, original size)
IMAGE_ROOT = '../data/images'
IMAGE_DEST = '../data/resized_images'

ims = glob.glob(os.path.join(IMAGE_ROOT, '*.jpg'))

counter = 0
destnames = []
for filename in ims:
    im = cv2.imread(filename)
    if im != None:
        small = cv2.resize(im, (500,500))
    	im_name = str.split(filename, '/')[-1]
        destname = os.path.join(IMAGE_DEST, im_name)
    	cv2.imwrite(destname, small)
        destnames.append(destname)
        #print cv2.imwrite(im_name, small)
    	counter += 1
    	if counter % 100 == 0:
    		print '%d images done...' % counter

with open('../cnnembed/val_imgs_med.txt', 'wb') as f:
    for name in destnames:
        #print uid
        f.write(name + '\n')

print 'done'
