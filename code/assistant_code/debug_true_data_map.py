#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:01:43 2018

@author: yuyi
"""

import numpy as np

import sys
sys.path
sys.path.append('/home/yuyi/Desktop/detect/code')
import utili as u




result_path = '/home/yuyi/Desktop/detect/output/pre_gt_result.npy' 
(gt_bboxes_all,gt_class_all,pre_bboxes_all,pre_class_all,pre_score_all) = np.load(result_path)



prediction = (pre_score_all,pre_class_all,pre_bboxes_all)
gt = (gt_class_all,gt_bboxes_all)
yolo_map = u.mAP(prediction,gt,20,0.5)



