# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:27:17 2018

@author: yuyi
"""
import tensorflow as tf
import config as cfg
import numpy as np
import cv2
import xml.etree.ElementTree as ET




################ used for construct the backbone ###################################

def conv_layer(x,k_size,name='0_conv',batch_norm=True,active=True,is_trainning=True):
    '''
        conv_layer
        输入参数:
            x:输入的张量[batch,w,h,channal]
            k_size：卷积核大小[1,k_w,k_h,1]
            name:卷积层的名字
            batch_norm:是否用batch_norm
            is_trainning: 训练还是测试
        输出：
            conv_out:卷积后的张量[batch,w,h,channal]
            
    '''

    # 卷积层,有BN层就不需要偏置项b
    w = tf.Variable(tf.truncated_normal(k_size,mean=0.0,stddev=1e-3),name = 'weight')
    conv_x = tf.nn.conv2d(x,w,[1,1,1,1],padding='SAME',name = name)
    out =  conv_x
    # BN层
    if batch_norm:
        out = tf.layers.batch_normalization(conv_x,axis=-1,training=is_trainning)
    # 激活层
    if active:
        out = tf.nn.leaky_relu(out,alpha=cfg.leaky_relu_alpha)
        
    return out

def pooling_layer(x,name='0_pooling'):
    with tf.name_scope(name):       
        pool = tf.nn.max_pool(x,[1,3,3,1],[1,2,2,1],padding='SAME',name = name)
    return pool

def reorg(x,name='reorg_x'):
    with tf.name_scope(name):
        output_1 = x[:,::2,::2,:]
        output_2 = x[:,::2,1::2,:]
        output_3 = x[:,1::2,1::2,:]
        output_4 = x[:,1::2,::2,:]
        output = tf.concat([output_1,output_2,output_3,output_4],axis=3)
    return output

##################################################################################################
    

###########################used for prediction and testing phase ########################################

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
        
        
        self.debug = []
        
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
            sorted_indx = np.argsort(pre_score_i,axis=0).squeeze()
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
        union = np.maximum(0,diff[:,0])*np.maximum(0,diff[:,1])
            
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


def read_im(image_path):
    im_ori = cv2.imread(image_path)
    h,w = im_ori.shape[0],im_ori.shape[1]
    im_resize = cv2.resize(im_ori,(cfg.image_size,cfg.image_size))
    im_resize = cv2.cvtColor(im_resize,cv2.COLOR_BGR2RGB).astype(np.float32)
    im_ori = cv2.cvtColor(im_ori,cv2.COLOR_BGR2RGB)
    im_resize = im_resize/256.0
    
    
    im_resize = np.reshape(im_resize,(1,cfg.image_size,cfg.image_size,3))
    return im_resize,im_ori,h,w

def read_test_xml(xml_path,w,h):
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    gt_boxes = []
    gt_class = []

    for i,obj in enumerate(objs):
        box = obj.find('bndbox')
        x_min = min(max(float(box.find('xmin').text),0),w)
        y_min = min(max(float(box.find('ymin').text),0),h)
        x_max = min(max(float(box.find('xmax').text),0),w)
        y_max = min(max(float(box.find('ymax').text),0),h)
        cls = obj.find('name').text
        gt_boxes.append([x_min,y_min,x_max,y_max])
        gt_class.append(cls)
        
    
    return gt_class,gt_boxes


# 计算两个box的IOU
def bboxes_iou(bboxes1,bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])
    
    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax-int_ymin,0.)
    int_w = np.maximum(int_xmax-int_xmin,0.)
    
    # 计算IOU
    int_vol = int_h * int_w # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
    IOU = int_vol / (vol1 + vol2 - int_vol) # IOU=交集/并集
    return IOU



def nms(obj_bbox,obj_class,obj_score):
    keep_bboxes = np.ones(obj_score.shape,dtype=np.bool)
    for i in range(obj_score.size-1):
        if keep_bboxes[i]:
            overlap = bboxes_iou(obj_bbox[i],obj_bbox[i+1:])
            keep_overlap = np.logical_or(overlap<cfg.threshold_nms,obj_class[i+1:]!=obj_class[i])
            keep_bboxes[i+1:] = np.logical_and(keep_overlap,keep_bboxes[i+1:])
            
    index = np.where(keep_bboxes)
    return (obj_bbox[index[0]],obj_class[index[0]],obj_score[index[0]])

def cut_bbox(obj_box,w,h):
    obj_box = np.transpose(obj_box)
    obj_box[0] = np.maximum(obj_box[0],0)
    obj_box[1] = np.maximum(obj_box[1],0)
    obj_box[2] = np.minimum(obj_box[2],w)
    obj_box[3] = np.minimum(obj_box[3],h)
    obj_box = np.transpose(obj_box)
    return obj_box

def postprosessing(obj_value,class_value,bboxes_value,w,h):
    obj_value = np.squeeze(obj_value)
    class_value = np.squeeze(class_value)
    bboxes_value = np.squeeze(bboxes_value)
    

    # reshape
    obj_value = np.reshape(obj_value,(-1,1))
    class_value = np.reshape(class_value,(-1,cfg.num_class))
    bboxes_value = np.reshape(bboxes_value,(-1,4))
    
    
    #threshold some box that is not contain object
    class_prob = class_value*obj_value
    index_obj = np.where(np.max(class_prob,axis=1)>cfg.threshold_obj)
    
    obj_class = np.argmax(class_prob[index_obj[0]],axis=-1)
    obj_score = class_prob[index_obj[0],obj_class]
    obj_bbox = bboxes_value[index_obj[0]]
    
    # decode the bbox
    obj_bbox[:,0] = obj_bbox[:,0]*w
    obj_bbox[:,1] = obj_bbox[:,1]*h
    obj_bbox[:,2] = obj_bbox[:,2]*w
    obj_bbox[:,3] = obj_bbox[:,3]*h
    # convert to xmin ymin xmax ymax
    obj_bbox[:,0],obj_bbox[:,1],obj_bbox[:,2],obj_bbox[:,3] = \
    obj_bbox[:,0]-obj_bbox[:,2]/2,obj_bbox[:,1]-obj_bbox[:,3]/2,\
    obj_bbox[:,0]+obj_bbox[:,2]/2,obj_bbox[:,1]+obj_bbox[:,3]/2
    # convert float to int
    obj_bbox = obj_bbox.astype(np.int32)
    obj_bbox = cut_bbox(obj_bbox,w,h)
    
    # sort score,class and bbox according score
    score_sort = np.argsort(obj_score)[::-1]
    obj_class = obj_class[score_sort]
    obj_score = obj_score[score_sort]
    obj_bbox = obj_bbox[score_sort]
    # NMS
    result  = nms(obj_bbox,obj_class,obj_score)
    
    return result

def draw_box(image_orig,result):
    # RBG -> BGR
    image_orig=cv2.cvtColor(image_orig,cv2.COLOR_RGB2BGR)
    
    (obj_bbox,obj_class,obj_score) = result
    h,w,_ = image_orig.shape
    thick = int((h+w)/300)
    for i,box in enumerate(obj_bbox):
        cv2.rectangle(image_orig,(box[0],box[1]),(box[2],box[3]),(0,0,255),thick)
        mess = '%s:%.3f'%(cfg.classes[obj_class[i]],obj_score[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(image_orig,mess,text_loc,cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
    return image_orig    
    
