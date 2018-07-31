# -*- coding: utf-8 -*-
"""
Created on Sat May 26 22:23:16 2018

@author: yuyi
"""




leaky_relu_alpha = 0.1
box_per_cell = 5
num_class = 20
num_predict = box_per_cell*(num_class+5)

image_size = 416

anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]



object_scale = 5
noobeject_scale = 1
coordi_scale = 5
class_scale = 1



loss_parameter = ()


exclude_variable = ['batch','weight_22:0']
#exclude_variable = ['']

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

class2ind = dict(zip(classes,range(len(classes))))
ind2class = dict(zip(range(len(classes)),classes))



cell_size = 13


# Trainning parameter

batch_size = 16
epoch = 50
saver_iter = 5000
summary_iter = 20
learning_rate = 0.0001

# train path

data_path = '/home/yuyi/Desktop/detect/data/VOCdevkit/VOC2012'
record_path = '/home/yuyi/Desktop/detect/data/record'
ini_weight_path = '/home/yuyi/Desktop/detect/output/init_weight/yolo_weights.ckpt'
out_path = '/home/yuyi/Desktop/detect/output'

test_path =  '/home/yuyi/Desktop/detect/test'

# test and prediction phase
threshold_obj = 0.5
threshold_nms = 0.5
model_path = '/home/yuyi/Desktop/detect/output/model_weight/yolo_v2-35000'
test_data_path = '/home/yuyi/Desktop/detect/data/VOCdevkit_test/VOC2007'



#mAP iou threshhold
map_thresh = 0.5







