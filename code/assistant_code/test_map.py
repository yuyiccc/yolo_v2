#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:52:28 2018

@author: yuyi
"""
import numpy as np
from mAP import mAP


# test calculate ap
#precision = np.array([1.  , 1.  , 0.67, 0.5 , 0.4 , 0.5 , 0.57, 0.5 , 0.44, 0.5 ])
#recall = np.array([0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 1. ])
#p = precision[0]
#for j in range(1,11):
#    try:
#        p += max(precision[np.where(recall>=j*0.1)[0][0]:])
#        print(max(precision[np.where(recall>=j*0.1)[0][0]:]))
#    except:
#        p += 0            
#ap  = p/11


#test _iou
#prediction  = np.zeros((1,4))
#prediction[0,:] = [50,50,100,100]
#gt = np.zeros((3,4))
#gt[0,:] = [60,60,100,100]
#gt[1,:] = [150,150,100,100]
#gt[2,:] = [50,50,60,60]
#
#test = mAP([1,1,1],[2,2],None,None)
#print(test._iou(prediction,gt))

#num_class = 2
#iou_thresh = 0.6
#
#pre_score = np.array([0.6,0.8,0.5,0.3])
#pre_class =  np.array([1,1,0,0])
#pre_box = np.zeros((4,4))
#pre_box[0,:] = [62,62,100,100]
#pre_box[2,:] = [62,62,100,100]
#pre_box[1,:] = [150,150,100,100]
#pre_box[3,:] = [50,50,60,60]
#pre = (pre_score,pre_class,pre_box)
#
#gt_class = np.array([1,1,0,0])
#gt_box = np.zeros((4,4))
#gt_box[0,:] = [65,63,100,100]
#gt_box[1,:] = [62,62,100,100]
#gt_box[2,:] = [152,156,100,100]
#gt_box[3,:] = [51,58,60,60]
#gt = (gt_class,gt_box)
#
#test = mAP(pre,gt,num_class,iou_thresh)
#result = test.out
#print(result)


#
## test _find_gt_index
#recall_flag = np.array([[0],[1],[0]])
#iou = np.array([[0.6],[0.8],[0.7]])
#thresh = 0.65
#
#
#def _find_gt_index(thresh,recall_flag,iou):
#    '''
#    find gt which's iou is biggest in recall_flag==0
#    recall_flag.shape:[m,1]
#    iou.shape:[m,1]
#    '''
#    oder = np.argsort(iou,axis=0)[::-1].squeeze()
#    indx = None
#    for i in oder:
#        if iou[i]<thresh:
#            break
#        elif recall_flag[i]==0:
#            indx = i
#            break
#        else:
#            continue
#    return indx
#
#
#print(_find_gt_index(thresh,recall_flag,iou))
