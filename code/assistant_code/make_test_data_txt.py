#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:06:51 2018

@author: yuyi
"""

import os
import glob
'''
there is no text.txt in ./ImageSets/Main path 
this scipt is to create a text.txt using all images names substrct train images names.


'''
train_txt = '/home/yuyi/Desktop/detect/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
image_path = '/home/yuyi/Desktop/detect/data/VOCdevkit/VOC2012/JPEGImages'
test_txt = '/home/yuyi/Desktop/detect/data/VOCdevkit/VOC2012/ImageSets/Main/test.txt'

#read all images names
files_path = glob.glob(image_path+'/*.jpg')
all_files_name = [os.path.basename(file).split('.')[0] for file in files_path]

# read all train images names
train_file_name = open(train_txt,'r').readlines()
train_file_name = [file.strip() for file  in train_file_name]

# select images names that are not in train data set
test_file_name = []
for name in all_files_name:
    if name not in  train_file_name:
        test_file_name.append(name+'\n')
        
# write those name to a file
with open(test_txt,'w') as f:
    f.writelines(test_file_name)
    
