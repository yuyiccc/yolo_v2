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
from utili import postprosessing,draw_box
from yolo_v2 import yolo_v2

from tensorflow.python import debug as tf_debug

w,h = 416,416


data = pascal_voc('train')

iterator,images,labes = data.build_data()


#model_path = '/home/yuyi/Desktop/detect/output/model_weight/yolo_v2-35000'

#img = tf.placeholder(tf.float32,shape=[1,cfg.image_size,cfg.image_size,3])
#label = tf.placeholder(tf.float32,shape=[1,cfg.cell_size,cfg.cell_size,5,25])
yolo_model = yolo_v2(images,labels=None,is_trainning=False)
#loss = yolo_model.total_loss
obj_probs,class_probs,bboxes_probs = yolo_model.predict
saver = tf.train.Saver()

with tf.Session() as sess:
#    if(cfg.debug):
#        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    saver.restore(sess,cfg.model_path)
    for i in range(30):
        im,la = sess.run((images,labes))
        im_o = np.squeeze(im)
        im_o = 256.0*im_o
        im_o = im_o.astype(np.uint8)
        obj_value,class_value,bboxes_value = sess.run((obj_probs,class_probs,bboxes_probs))


        result = postprosessing(obj_value,class_value,bboxes_value,w,h)
        img_detection = draw_box(im_o,result)
        
        cv2.imwrite("detection.jpg", img_detection)
        cv2.imshow("detection_results", img_detection)
        cv2.waitKey(0)
        
        
        
        
        im_la = np.squeeze(im)
        im_la = 256.0*im_la
        im_la = im_la.astype(np.uint8)
        #cv2.imshow('im',im)
        #cv2.waitKey(0)
        
        bboxes_label = la[:,:,:,:,1:5]
        obj_label = la[:,:,:,:,0]
        obj_label = np.expand_dims(obj_label,axis=-1)
        class_label =la[:,:,:,:,5:]
        
        
        result = postprosessing(obj_label,class_label,bboxes_label,w,h)
        im_detection = draw_box(im_la,result)
        cv2.imshow('im_la',im_detection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


tf.reset_default_graph()

