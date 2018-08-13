# -*- coding: utf-8 -*-
"""
Created on Sat May 26 22:23:16 2018

@author: yuyi
"""


anchors = [[1.3221, 1.73145],
           [3.19275, 4.00944],
           [5.05587, 8.09892],
           [9.47112, 4.84053],
           [11.2364, 10.0071]]

leaky_relu_alpha = 0.1
box_per_cell = len(anchors)
num_class = 20
num_predict = box_per_cell*(num_class+5)

image_size = 416
eps = 1e-10




object_scale = 5
noobject_scale = 1
#use this ratio to control of no object cell's number
noobject_ratio = 0.7
coordi_scale = 1
class_scale = 1


debug = True
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

batch_size = 1
epoch = 50
saver_iter = 5000
summary_iter = 20
learning_rate = 0.0005

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







