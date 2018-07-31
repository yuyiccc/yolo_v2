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
        self.image_size = cfg.image_size
        
        self.object_scale = 5
        self.noobeject_scale = 1
        self.coordi_scale = 5
        self.class_scale = 1
        
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
        
#        num_anchor = len(self.anchors)
        anchors = tf.constant(self.anchors,dtype=tf.float32)
        anchors_w = tf.reshape(anchors[:,0],shape=[1,1,self.box_per_cell])
        anchors_h = tf.reshape(anchors[:,1],shape=[1,1,self.box_per_cell])
        
        obj_probs = tf.nn.sigmoid(predict[:,:,:,0])
        xy_offset = tf.nn.sigmoid(predict[:,:,:,1:3])
        wh_offset = tf.exp(predict[:,:,:,3:5])
        class_probs = tf.nn.softmax(predict[:,:,:,5:])
        
        hight_index = tf.range(self.cell_size,dtype=tf.float32)
        weight_index = tf.range(self.cell_size,dtype=tf.float32)
        
        x_cell,y_cell = tf.meshgrid(weight_index,hight_index)
        x_cell = tf.reshape(x_cell,shape=[1,-1,1])
        y_cell  = tf.reshape(y_cell,shape=[1,-1,1])
        
        box_x = (x_cell+xy_offset[:,:,:,0])/self.cell_size
        box_y = (y_cell+xy_offset[:,:,:,1])/self.cell_size
        box_w = tf.sqrt((anchors_w*wh_offset[:,:,:,0])/self.cell_size)
        box_h = tf.sqrt((anchors_h*wh_offset[:,:,:,1])/self.cell_size)
        
        bboxes_probs = tf.stack([box_x,box_y,box_w,box_h],axis=3)
        
        return (obj_probs,class_probs,bboxes_probs)
        
    
    
    
    def loss(self,predict,labels):
        obj_probs,class_probs,bboxes_probs = predict
        
        bboxes_probs = tf.reshape(bboxes_probs,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,4])
        class_probs = tf.reshape(class_probs,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,self.num_class])
        obj_probs = tf.reshape(obj_probs,shape=[-1,self.cell_size,self.cell_size,self.box_per_cell,1])
        
        bboxes_label = labels[:,:,:,:,1:5]
        obj_label = labels[:,:,:,:,0]
        obj_label = tf.expand_dims(obj_label,axis=-1)
        class_label =labels[:,:,:,:,5:]
        
        iou = self.cal_iou(bboxes_label,bboxes_probs)
        best_anchor = tf.cast(tf.equal(iou,tf.reduce_max(iou,axis=-1,keepdims=True)),dtype=tf.float32)
        # 标记别分配的anchor
        with_object_mask = tf.expand_dims(best_anchor*obj_label, axis = 4)
        # object loss 的mask，包括了背景类
        object_mask = tf.squeeze(self.object_scale*with_object_mask+self.noobeject_scale*(1.0-with_object_mask),axis=-1)
        coord_mask = tf.squeeze(self.coordi_scale*with_object_mask,axis=-1)
        class_mask = tf.squeeze(self.class_scale*with_object_mask,axis=-1)
        
        object_loss = object_mask*tf.square(obj_label-obj_probs)
        coord_loss = coord_mask*tf.square(bboxes_label-bboxes_probs)
        class_loss = class_mask*tf.square(class_label-class_probs)
        
        object_loss = tf.reduce_mean(tf.reduce_sum(object_loss,axis=[1,2,3,4]))
        coord_loss  = tf.reduce_mean(tf.reduce_sum(coord_loss,axis=[1,2,3,4]))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss,axis=[1,2,3,4]))
        loss = tf.add_n([object_loss,coord_loss,class_loss])
        tf.summary.scalar('object_loss',object_loss)
        tf.summary.scalar('coord_loss',coord_loss)
        tf.summary.scalar('class_loss',class_loss)
        
#        loss = tf.concat([object_loss,coord_loss,class_loss],axis=-1)
#        loss = tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2,3,4]),name='total_loss')
        
        return loss
    
    def cal_iou(self,box_1,box_2):
        boxx = tf.square(box_1[:,:,:,:,2:4])*self.image_size
        box_1_squre = boxx[:,:,:,:,0]*boxx[:,:,:,:,1]
        
        boxes_1 = tf.stack([box_1[:,:,:,:,0]*self.image_size-0.5*boxx[:,:,:,:,0],
                            box_1[:,:,:,:,1]*self.image_size-0.5*boxx[:,:,:,:,1],
                            box_1[:,:,:,:,0]*self.image_size+0.5*boxx[:,:,:,:,0],
                            box_1[:,:,:,:,1]*self.image_size+0.5*boxx[:,:,:,:,1]],axis=4)
    
        boxx = tf.square(box_2[:,:,:,:,2:4])*self.image_size
        box_2_squre = boxx[:,:,:,:,0]*boxx[:,:,:,:,1]
        
        boxes_2 = tf.stack([box_2[:,:,:,:,0]*self.image_size-0.5*boxx[:,:,:,:,0],
                            box_2[:,:,:,:,1]*self.image_size-0.5*boxx[:,:,:,:,1],
                            box_2[:,:,:,:,0]*self.image_size+0.5*boxx[:,:,:,:,0],
                            box_2[:,:,:,:,1]*self.image_size+0.5*boxx[:,:,:,:,1]],axis=4)
    
    
        left_up = tf.maximum(boxes_1[:,:,:,:,0:2],boxes_2[:,:,:,:,0:2])
        right_down = tf.maximum(boxes_1[:,:,:,:,2:4],boxes_2[:,:,:,:,2:4])
        intersection = tf.maximum(right_down-left_up,0.0)
        inter_squre = intersection[:,:,:,:,0]*intersection[:,:,:,:,1]
        union_squre = box_1_squre+box_2_squre-inter_squre
        
        return tf.expand_dims(tf.clip_by_value(1.0*inter_squre/union_squre,0.0,1.0),axis=-1)
