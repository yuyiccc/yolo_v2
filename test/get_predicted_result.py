#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:02:25 2018

@author: yuyi
"""

import tensorflow as tf
import os

import sys 
sys.path.insert(0,'/home/yuyi/Desktop/detect/code')


from yolo_v2 import yolo_v2
import utili as u
import config as cfg
from shutil import copyfile

class cal_mAP:  
    def __init__(self):
        self.main()
    def main(self):
        im = tf.placeholder(dtype=tf.float32,shape=[1,cfg.image_size,cfg.image_size,3],name='input_image')
#        w = tf.placeholder(dtyep=tf.float32,shape=[],name='im_w')
#        h = tf.placeholder(dtype=tf.float32,shape=[],name='im_h')
        
        yolo = yolo_v2(im,is_trainning=False)
        obj_probs,class_probs,bboxes_probs = yolo.predict
        
        im_path_list,xml_path_list =  self.glob_im_xml_path()
        
        saver = tf.train.Saver()
        

        prediciton_path  = os.path.join(cfg.test_path,'predicted')
        if not os.path.isdir(prediciton_path):
            os.mkdir(prediciton_path)
        
        gt_path  = os.path.join(cfg.test_path,'ground-truth')
        if not os.path.isdir(gt_path):
            os.mkdir(gt_path)
        
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,cfg.model_path)
            
            num_image = len(im_path_list)
            for i,(im_file,xml_file) in enumerate(zip(im_path_list,xml_path_list)):
            
                # get image name for save the result
                # read the image
                im_name = os.path.basename(im_file)
                
                #copy image 
                copyfile(im_file,os.path.join(cfg.test_path,'images',im_name))
                
                im_resize,im_orig,h,w = u.read_im(im_file)
                # read xml file to get gt box and write to the txt
                gt_class,gt_bboxes = u.read_test_xml(xml_file,w,h)
                with open(os.path.join(gt_path,im_name[:-4]+'.txt'),'w') as f:
                    for class_i,bbox_i in zip(gt_class,gt_bboxes):
                        f.writelines('%s %.2f %.2f %.2f %.2f\n'%(class_i,bbox_i[0],bbox_i[1],bbox_i[2],bbox_i[3]))
                
                
                
                # get the prediction
                pre_score_i,pre_class_i,pre_bboxes_i = sess.run((obj_probs,class_probs,bboxes_probs),feed_dict={im:im_resize})               
                # post prosessing the result
                pre_bbox,pre_class,pre_score = u.postprosessing(pre_score_i,pre_class_i,pre_bboxes_i,w,h)
                
                #save the results to the txt
                with open(os.path.join(prediciton_path,im_name[:-4]+'.txt'),'w') as f:
                    for bbox_i,class_i,score_i in zip(pre_bbox,pre_class,pre_score):
                        f.writelines('%s %.4f %.2f %.2f %.2f %.2f\n'%(cfg.ind2class[class_i],score_i,bbox_i[0],bbox_i[1],bbox_i[2],bbox_i[3]))                
                
                if i%50==0:
                    print('----complete :%d/%d-----'%(i,num_image))
            

        
    def glob_im_xml_path(self):
        '''
        return test image's path list by read test.txt
        '''
        # find the test.txt's path and read it
        test_im_txt = os.path.join(cfg.test_data_path,'ImageSets','Main','test.txt')
        name_list = open(test_im_txt).readlines()
        # remove \n and add .jpg
        im_path_list = [name.strip('\n')+'.jpg' for name in name_list]
        xml_path_list = [name.strip('\n')+'.xml' for name in name_list]
        # add the abs path
        im_abs_path = os.path.join(cfg.test_data_path,'JPEGImages')
        xml_abs_path = os.path.join(cfg.test_data_path,'Annotations')
        
        im_path_list = [os.path.join(im_abs_path,name)for name in im_path_list]
        xml_path_list = [os.path.join(xml_abs_path,name)for name in xml_path_list]
        
        return im_path_list,xml_path_list
if __name__=='__main__':
    cal_mAP()