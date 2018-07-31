#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:39:09 2018

@author: yuyi
"""
import numpy as np


class mAP:
    '''
    #calulating the mAP of one detector
    '''
    def __init__(self,Prediction,GT,Num_class,iou_thresh):
        '''
        Prediction is detector's output which is include probility of objects,class of objects and bounding box of objects.
        GT is Ground truth which is include class of real objects and bounding box of real objects.
        Num_class:(type :int) number of classs
        iou_thresh(tyep:float)biger than iou_thresh is means the gt box is being detected
        '''
        #pre_score.shape:[N,1] , pre_class.shape:[N,1] , pre_bbox.shape:[N,4]
        #class should be start from zero
        self.pre_score,self.pre_class,self.pre_bbox = Prediction
        #gt_class.shape:[N,1] , pgt_bbox.shape:[N,4]
        self.gt_class,self.gt_bbox = GT
        self.num_class =Num_class
        self.thresh = iou_thresh
        self.out = self._cal_map()

    def _cal_map(self): 
        out_put = {}
        sum_ap = 0
        #calculate every class's AP
        for i in range(self.num_class):
            
            # find i-th gt object's bbox
            gt_indx_i = np.where(self.gt_class==i)[0]
            gt_bbox_i = self.gt_bbox[gt_indx_i]
            
            # find i-th predict object's bbox and score
            pre_indx_i = np.where(self.pre_class==i)[0]
            pre_score_i  = self.pre_score[pre_indx_i]
            pre_bbox_i = self.pre_bbox[pre_indx_i]
            
            
            # sort the prediction's bbox from hight to low acrodding to score
            sorted_indx = np.argsort(pre_score_i).squeeze()
            pre_bbox_i = pre_bbox_i[sorted_indx]
            
            AP_i =self._cal_ap(pre_bbox_i,gt_bbox_i)
            
            out_put["class_%s"%i] = AP_i
            sum_ap += AP_i
        out_put["mAP"] = sum_ap/self.num_class 
        return out_put        
                
    def _cal_ap(self,pre_bbox,gt_bbox):
        # number of predcition box and gt box
        # pre_bbox.shape:[n,4]
        # gt_bbox.shape:[m,4]
        
        n = pre_bbox.shape[0]
        m = gt_bbox.shape[0]
        recall_flag = np.zeros((m,1))
        
        precision = np.zeros((n,1))
        recall = np.zeros((n,1))
        
        
        TP = 0
        for i in range(n):
            iou = self._iou(pre_bbox[i,:],gt_bbox)
            gt_indx = self._find_gt_index(recall_flag,iou)
#            gt_indx = np.argmax(iou,axis=0)
#            
#            #if the gt box is already detected by prior prediction box or iou is smaller than thresh then TP will not +1
#            if recall_flag[gt_indx]==1 or iou[gt_indx]<self.thresh:
            if gt_indx==None:
                precision[i] = TP/(i+1)
                recall[i] = TP/m
            else:
                recall_flag[gt_indx]=1
                TP += 1
                precision[i] = TP/(i+1)
                recall[i] = TP/m
        
        #calculate ap from recall and precision
        p = np.copy(precision[0])
        #for debug
        #print(precision)
        #print(recall)
        for j in range(1,11):
            try:
                p += max(precision[np.where(recall>=j*0.1)[0][0]:])
                #for debug
                #print(j,max(precision[np.where(recall>=j*0.1)[0][0]:]))
            except:
                p += 0            
        ap  = p/11
        
        return ap
    
    
    def _iou(self,pre_bbox,gt_bbox):
        # pre_bbox.shape:[1,4]
        # gt_box.shape:[m,4]
        #[x,y,w,h]
        m = gt_bbox.shape[0]
        pre_bbox = np.tile(pre_bbox,[m,1])
        
        gt_area = gt_bbox[:,2]*gt_bbox[:,3]
        pre_area = pre_bbox[:,2]*pre_bbox[:,3]
        
        #transform (x,y,w,h)->(x_min,y_min,x_max,y_max)
        gt_bbox = np.stack(
                  [gt_bbox[:,0]-gt_bbox[:,2]/2,
                   gt_bbox[:,1]-gt_bbox[:,3]/2,
                   gt_bbox[:,0]+gt_bbox[:,2]/2,
                   gt_bbox[:,1]+gt_bbox[:,3]/2]
                  ).transpose((1,0))
        
        pre_bbox = np.stack(
          [pre_bbox[:,0]-pre_bbox[:,2]/2,
           pre_bbox[:,1]-pre_bbox[:,3]/2,
           pre_bbox[:,0]+pre_bbox[:,2]/2,
           pre_bbox[:,1]+pre_bbox[:,3]/2]
          ).transpose((1,0))
        
        left_up = np.maximum(gt_bbox[:,:2],pre_bbox[:,:2])
        right_down = np.minimum(gt_bbox[:,2:],pre_bbox[:,2:])
        
        diff = right_down-left_up
        union = np.maximum(0,diff[:,0]*diff[:,1])
        
        iou  = union/(gt_area+pre_area-union)
        return iou
        
    def _find_gt_index(self,recall_flag,iou):
        '''
        find gt which's iou is biggest in recall_flag==0
        recall_flag.shape:[m,1]
        iou.shape:[m,1]
        '''
        oder = np.argsort(iou,axis=0)[::-1].squeeze()
        indx = None
        for i in oder:
            if iou[i]<self.thresh:
                break
            elif recall_flag[i]==0:
                indx = i
                break
            else:
                continue
        return indx
        