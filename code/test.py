#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:02:25 2018

@author: yuyi
"""

import tensorflow as tf
from yolo_v2 import yolo_v2
import os
import cv2
import numpy as np




import utili as u
import config as cfg


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
        
        # save result path
        result_path = os.path.join(cfg.out_path,'test_result')
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        
        
        # save the prediction's result
        pre_bboxes_all = np.zeros((1,4))
        pre_class_all = np.zeros((1,1))
        pre_score_all = np.zeros((1,1))
        # save gt result
        gt_bboxes_all = np.zeros((1,4))
        gt_class_all = np.zeros((1,1))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,cfg.model_path)
            
            num_image = len(im_path_list)
            for i,(im_file,xml_file) in enumerate(zip(im_path_list,xml_path_list)):
                
                # get image name for save the result
                # read the image
                im_name = os.path.basename(im_file)
                im_resize,im_orig,h,w = u.read_im(im_file)
                # read xml file to get gt box
                gt_class_i,gt_bboxes_i = u.read_test_xml(xml_file,w,h)
                gt_class_all = np.concatenate((gt_class_all,gt_class_i))
                gt_bboxes_all = np.concatenate((gt_bboxes_all,gt_bboxes_i))
                # get the prediction
                pre_score_i,pre_class_i,pre_bboxes_i = sess.run((obj_probs,class_probs,bboxes_probs),feed_dict={im:im_resize})               
                # post prosessing the result
                pre_bbox_i,pre_class_i,pre_score_i = u.postprosessing(pre_score_i,pre_class_i,pre_bboxes_i,w,h)
                # draw the detection result to the image
                img_detection = u.draw_box(im_orig,(pre_bbox_i,pre_class_i,pre_score_i))
                #save the result image
                result_im_file = os.path.join(result_path,im_name)
                cv2.imwrite(result_im_file,img_detection)
                # save the detection result for caculate the mAP
                pre_bboxes_all = np.concatenate((pre_bboxes_all,pre_bbox_i))
                pre_class_all = np.concatenate((pre_class_all,np.expand_dims(pre_class_i,axis=1)))
                pre_score_all = np.concatenate((pre_score_all,np.expand_dims(pre_score_i,axis=1)))
                if i%50==0:
                    print('----complete :%d/%d-----'%(i,num_image))
            
            #
            pre_bboxes_all = pre_bboxes_all[1:]
            pre_class_all = pre_class_all[1:]
            pre_score_all = pre_score_all[1:]
            gt_bboxes_all = gt_bboxes_all[1:]
            gt_class_all = gt_class_all[1:]
            # transform (xmin,ymin,xmax,ymax)->(x,y,w,h)
            gt_bboxes_all = self.transform(gt_bboxes_all)
            pre_bboxes_all = self.transform(pre_bboxes_all)
            
            # save the prediction and gt result to the disk
            pre_gt_result_file = os.path.join(cfg.out_path,'pre_gt_result.npy')
            np.save(pre_gt_result_file,(gt_bboxes_all,gt_class_all,pre_bboxes_all,pre_class_all,pre_score_all))
            
            # caculate the mAP
            prediction = (pre_score_all,pre_class_all,pre_bboxes_all)
            gt = (gt_class_all,gt_bboxes_all)
            yolo_map = u.mAP(prediction,gt,cfg.num_class,cfg.map_thresh)
            print(yolo_map.out)

                
                
                
    def transform(self,bbox):
        '''
        transform (xmin,ymin,xmax,ymax)->(x,y,w,h)
        '''
        
        re_bbox = np.stack([(bbox[:,0]+bbox[:,2])/2,
                            (bbox[:,1]+bbox[:,3])/2,
                            (bbox[:,2]-bbox[:,0]),
                            (bbox[:,3]-bbox[:,1])
                            ]).transpose()
        return re_bbox
        
        
        
        
    def glob_im_xml_path(self):
        '''
        return test image's path list by read test.txt
        '''
        # find the test.txt's path and read it
        test_im_txt = os.path.join(cfg.data_path,'ImageSets','Main','test.txt')
        name_list = open(test_im_txt).readlines()
        # remove \n and add .jpg
        im_path_list = [name.strip('\n')+'.jpg' for name in name_list]
        xml_path_list = [name.strip('\n')+'.xml' for name in name_list]
        # add the abs path
        im_abs_path = os.path.join(cfg.data_path,'JPEGImages')
        xml_abs_path = os.path.join(cfg.data_path,'Annotations')
        
        im_path_list = [os.path.join(im_abs_path,name)for name in im_path_list]
        xml_path_list = [os.path.join(xml_abs_path,name)for name in xml_path_list]
        
        return im_path_list,xml_path_list
if __name__=='__main__':
    cal_mAP()