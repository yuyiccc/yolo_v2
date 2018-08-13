# -*- coding: utf-8 -*-
"""
Created on Sat May 26 22:06:28 2018

@author: yuyi
"""
from utili import conv_layer,pooling_layer,reorg
import tensorflow as tf
import config as cfg




class yolo_v2():
    def __init__(self,images,labels=None,is_trainning=True):
        
        self.cell_size = cfg.cell_size
        self.box_per_cell = cfg.box_per_cell
        self.num_class = cfg.num_class
        self.anchors = cfg.anchors
        tensor_anchors = tf.constant(self.anchors,dtype=tf.float32,)
        self.anchors_w = tf.reshape(tensor_anchors[:,0],shape=[1,1,self.box_per_cell],)
        self.anchors_h = tf.reshape(tensor_anchors[:,1],shape=[1,1,self.box_per_cell],)
        self.image_size = cfg.image_size
        
        self.object_scale = cfg.object_scale
        self.noobject_scale = cfg.noobject_scale
        self.coordi_scale = cfg.coordi_scale
        self.class_scale = cfg.class_scale
        
        self.is_trainning  = is_trainning
        # num_cell*(num_class+5)
        self.num_predict = cfg.num_predict
        
        self.logits = self.build_network(images)
        self.predict = self.decode_logits()
        if is_trainning:
            self.total_loss = self.loss(self.predict,labels)
            tf.summary.scalar('total_loss',self.total_loss)
        
    def build_network(self,images):
        
        net = conv_layer(images, [3,3,3,32], name='0_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = pooling_layer(net, name = '1_pooling')
        
        net = conv_layer(net,[3,3,32,64],name = '2_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = pooling_layer(net,name = '3_pooling')
        
        net = conv_layer(net,[3,3,64,128],name = '4_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[1,1,128,64],name = '5_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[3,3,64,128],name = '6_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = pooling_layer(net,name = '7_pooling')
        
        net = conv_layer(net,[3,3,128,256],name = '8_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[1,1,256,128],name = '9_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[3,3,128,256],name = '10_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = pooling_layer(net,name = '11_pooling')
        
        net = conv_layer(net,[3,3,256,512],name = '12_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[1,1,512,256],name = '13_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[3,3,256,512],name = '14_conv', batch_norm = True, is_trainning = self.is_trainning)  
        net = conv_layer(net,[1,1,512,256],name = '15_conv', batch_norm = True, is_trainning = self.is_trainning)
        net_16 = conv_layer(net,[3,3,256,512],name = '16_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = pooling_layer(net_16,name = '17_pooling')
        
        net = conv_layer(net,[3,3,512,1024],name = '18_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[1,1,1024,512],name = '19_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[3,3,512,1024],name = '20_conv', batch_norm = True, is_trainning = self.is_trainning)  
        net = conv_layer(net,[1,1,1024,512],name = '21_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[3,3,512,1024],name = '22_conv', batch_norm = True, is_trainning = self.is_trainning)
        
        net = conv_layer(net,[3,3,1024,1024],name = '23_conv', batch_norm = True, is_trainning = self.is_trainning)
        net_32 = conv_layer(net,[3,3,1024,1024],name = '24_conv', batch_norm = True, is_trainning = self.is_trainning)
        
        net = conv_layer(net_16,[1,1,512,64],name = '25_conv', batch_norm = True, is_trainning = self.is_trainning)
        net  = reorg(net)
        net = tf.concat([net,net_32],axis=3)
        
        net = conv_layer(net,[3,3,int(net.get_shape()[-1]),1024],name = '26_conv', batch_norm = True, is_trainning = self.is_trainning)
        net = conv_layer(net,[1,1,1024,self.num_predict],name = 'logits', batch_norm = False, active=False,is_trainning = self.is_trainning)
        
        
        return net
    
    def decode_logits(self):
        out_shape = [-1,self.cell_size*self.cell_size,self.box_per_cell,self.num_class+5]#[~,13*13,5,25]
        predict = tf.reshape(self.logits,out_shape)
        
        
        obj_probs = tf.nn.sigmoid(predict[:,:,:,0])
        xy_offset = tf.nn.sigmoid(predict[:,:,:,1:3])
        wh_offset = predict[:,:,:,3:5]
        class_probs = tf.nn.softmax(predict[:,:,:,5:])
        
        
        hight_index = tf.range(self.cell_size,dtype=tf.float32)
        weight_index = tf.range(self.cell_size,dtype=tf.float32)
        
        x_cell,y_cell = tf.meshgrid(weight_index,hight_index)
        x_cell = tf.reshape(x_cell,shape=[1,-1,1])
        y_cell  = tf.reshape(y_cell,shape=[1,-1,1])
        
        box_x = (x_cell+xy_offset[:,:,:,0])/self.cell_size
        box_y = (y_cell+xy_offset[:,:,:,1])/self.cell_size
        box_w = (self.anchors_w*(tf.exp(wh_offset[:,:,:,0])-cfg.eps))/self.cell_size
        box_h = (self.anchors_h*(tf.exp(wh_offset[:,:,:,1])-cfg.eps))/self.cell_size
        
        bboxes_probs = tf.stack([box_x,box_y,box_w,box_h],axis=3)
        if self.is_trainning:
            bboxes_probs_logit = tf.concat([xy_offset,wh_offset],axis=3,name='coord_logit')
            return (obj_probs,class_probs,bboxes_probs,bboxes_probs_logit)
        else:
            return (obj_probs,class_probs,bboxes_probs)
        
    
    def encode_gtbox(self,boxes):
        '''
        boxes:[b,13,13,5,4]
        '''

        x = boxes[:,:,:,:,0]*self.cell_size
        y = boxes[:,:,:,:,1]*self.cell_size
        w = boxes[:,:,:,:,2]*self.cell_size
        h = boxes[:,:,:,:,3]*self.cell_size
        ind_x = tf.floor(x)
        ind_y = tf.floor(y)
        en_x = x-ind_x
        en_y = y-ind_y
        
        anchors_h = tf.reshape(self.anchors_h,shape=[1,1,1,self.box_per_cell])
        anchors_w = tf.reshape(self.anchors_w,shape=[1,1,1,self.box_per_cell])
        
        en_w = tf.log(w/anchors_w+cfg.eps,name='en_w')
        en_h = tf.log(h/anchors_h+cfg.eps,name='en_h')
        
        encode_bboxes = tf.stack([en_x,en_y,en_w,en_h],axis=4)
        
        return encode_bboxes
    
    def loss(self,predict,labels):
        obj_probs,class_probs,bboxes_probs,bboxes_probs_logit = predict
        
        # prediction
        bboxes_probs_logit = tf.reshape(bboxes_probs_logit,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,4])
        bboxes_probs = tf.reshape(bboxes_probs,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,4])
        class_probs = tf.reshape(class_probs,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,self.num_class])
        obj_probs = tf.reshape(obj_probs,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,1])
        
        
        # GT
        bboxes_label = labels[:,:,:,:,1:5]
        # encode_bbox
        encode_bboxes_label = self.encode_gtbox(bboxes_label)
        
        obj_label = labels[:,:,:,:,0]
        class_label =labels[:,:,:,:,5:]
        
        
        
        #iou's shape [b,13,13,5]
        #iou caculation only concern about w and h acoording to v2 papers
        bboxes_label_shift = tf.concat([bboxes_label[:,:,:,:,:2]*0,bboxes_label[:,:,:,:,2:]],axis=4)
        bboxes_prob_shift = tf.concat([bboxes_probs[:,:,:,:,:2]*0,bboxes_probs[:,:,:,:,2:]],axis=4)
        iou = self.cal_iou(bboxes_label_shift,bboxes_prob_shift)
        #***bug_2*** can't find the best anchor
        best_anchor = tf.cast(tf.equal(iou,tf.reduce_max(iou,axis=-1,keepdims=True)),dtype=tf.float32,name='best_anchor')
        # 标记别分配的anchor
        # anchor's and obj_label's shape are all [b,13,13,5or1 ]
        with_object_mask = tf.expand_dims(tf.multiply(best_anchor,obj_label),axis=-1,name='mask')
        
        
        
        
        
        sample_no_object = tf.random_uniform((cfg.batch_size,cfg.cell_size,cfg.cell_size,cfg.box_per_cell,1),0,1)
        sample_no_object = tf.cast(sample_no_object<cfg.noobject_ratio,dtype=tf.float32,name='sample')
        no_obeject_mask = tf.multiply((1.0-with_object_mask),sample_no_object,name='no_mask')

        object_mask = self.object_scale*with_object_mask+self.noobject_scale*no_obeject_mask
        # object loss 的mask，包括了背景类
#        object_mask = self.object_scale*with_object_mask+self.noobject_scale*(1.0-with_object_mask)
        coord_weight = tf.subtract(2.0,tf.expand_dims(bboxes_label[:,:,:,:,2]*bboxes_label[:,:,:,:,3],axis=-1),name='coord_weight')#(2-gt.w*gt.h)
        coord_mask = self.coordi_scale*coord_weight*with_object_mask
        class_mask = self.class_scale*with_object_mask
        

        
        obj_label = tf.expand_dims(obj_label,axis=-1,name='object_mask')
        
        # positive boxes's label is iou not 1
        iou_value = tf.expand_dims(iou,axis=-1)
        object_loss = tf.multiply(object_mask,tf.square(obj_label*iou_value-obj_probs),name='ob_l')

        coord_loss = tf.multiply(coord_mask,tf.square(encode_bboxes_label-bboxes_probs_logit),name='coor_l')
        class_loss = tf.multiply(class_mask,tf.square(class_label-class_probs),name='class_l')
        
        object_loss = tf.reduce_mean(tf.reduce_sum(object_loss,axis=[1,2,3,4]),name='object_loss')
        coord_loss  = tf.reduce_mean(tf.reduce_sum(coord_loss,axis=[1,2,3,4]),name='coord_loss')
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss,axis=[1,2,3,4]),name='class_loss')
        loss = tf.add_n([object_loss,coord_loss,class_loss])
        tf.summary.scalar('object_loss',object_loss)
        tf.summary.scalar('coord_loss',coord_loss)
        tf.summary.scalar('class_loss',class_loss)
        
#        loss = tf.concat([object_loss,coord_loss,class_loss],axis=-1)
#        loss = tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2,3,4]),name='total_loss')
        
        return loss
    
    def cal_iou(self,box_1,box_2):
        boxx = tf.multiply(box_1[:,:,:,:,2:4],self.image_size)
        box_1_squre = tf.multiply(boxx[:,:,:,:,0],boxx[:,:,:,:,1])
        
        boxes_1 = tf.stack([box_1[:,:,:,:,0]*self.image_size-0.5*boxx[:,:,:,:,0],
                            box_1[:,:,:,:,1]*self.image_size-0.5*boxx[:,:,:,:,1],
                            box_1[:,:,:,:,0]*self.image_size+0.5*boxx[:,:,:,:,0],
                            box_1[:,:,:,:,1]*self.image_size+0.5*boxx[:,:,:,:,1]],axis=4)
    
        boxx = tf.multiply(box_2[:,:,:,:,2:4],self.image_size,name='box2_wh')
        box_2_squre = tf.multiply(boxx[:,:,:,:,0],boxx[:,:,:,:,1],name='box2_squre')
        
        boxes_2 = tf.stack([box_2[:,:,:,:,0]*self.image_size-0.5*boxx[:,:,:,:,0],
                            box_2[:,:,:,:,1]*self.image_size-0.5*boxx[:,:,:,:,1],
                            box_2[:,:,:,:,0]*self.image_size+0.5*boxx[:,:,:,:,0],
                            box_2[:,:,:,:,1]*self.image_size+0.5*boxx[:,:,:,:,1]],axis=4)
    
    
        left_up = tf.maximum(boxes_1[:,:,:,:,0:2],boxes_2[:,:,:,:,0:2])
        right_down = tf.minimum(boxes_1[:,:,:,:,2:4],boxes_2[:,:,:,:,2:4])
        intersection = tf.maximum(right_down-left_up,0.0)
        inter_squre = tf.multiply(intersection[:,:,:,:,0],intersection[:,:,:,:,1])
        union_squre = tf.subtract(box_1_squre+box_2_squre,inter_squre)
        
#        return tf.clip_by_value(1.0*inter_squre/union_squre,0.0,1.0)
        return tf.divide(inter_squre,union_squre)
