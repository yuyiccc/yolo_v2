# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:54:17 2018

@author: Administrator
"""
import tensorflow as tf
from preprosess_data import pascal_voc
import numpy as np
import cv2
import config as cfg
from test import postprosessing,draw_box
from yolo_v2 import yolo_v2

w,h = 416,416


data = pascal_voc('train')

iterator,images,labes = data.build_data()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(6):
        im,la = sess.run((images,labes))


model_path = '/home/yuyi/Desktop/detect/output/model_weight/yolo_v2-12000'
im_o = np.squeeze(im)
im_o = 255*(im_o+1)/2
im_o = im_o.astype(np.uint8)
img = tf.placeholder(tf.float32,shape=[1,cfg.image_size,cfg.image_size,3])
label = tf.placeholder(tf.float32,shape=[1,cfg.cell_size,cfg.cell_size,5,25])
yolo_model = yolo_v2(img,label,is_trainning=False)
#loss = yolo_model.total_loss
obj_probs,class_probs,bboxes_probs = yolo_model.predict
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path)
    obj_value,class_value,bboxes_value = sess.run((obj_probs,class_probs,bboxes_probs),feed_dict={img:im})
#    obj_value,class_value,bboxes_value,loss_1 = sess.run((obj_probs,class_probs,bboxes_probs,loss),feed_dict={img:im,label:la})
result = postprosessing(obj_value,class_value,bboxes_value,w,h)
img_detection = draw_box(im_o,result)

cv2.imwrite("detection.jpg", img_detection)
cv2.imshow("detection_results", img_detection)
cv2.waitKey(0)




im = np.squeeze(im)
im = 255*(im+1)/2
im = im.astype(np.uint8)
#cv2.imshow('im',im)
#cv2.waitKey(0)

bboxes_label = la[:,:,:,:,1:5]
obj_label = la[:,:,:,:,0]
obj_label = np.expand_dims(obj_label,axis=-1)
class_label =la[:,:,:,:,5:]


result = postprosessing(obj_label,class_label,bboxes_label,w,h)
im_detection = draw_box(im,result)
cv2.imshow('im',im_detection)
cv2.waitKey(0)

tf.reset_default_graph()

