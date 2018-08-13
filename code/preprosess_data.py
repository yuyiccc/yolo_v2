# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:03:32 2018

@author: yuyi
"""

import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import config as cfg
import cv2
import numpy as np

class pascal_voc():
    def __init__(self, phase = 'train'):
        self.batch_size = cfg.batch_size
        self.epoch = cfg.epoch
        
        self.phase = phase
        self.record_path = cfg.record_path
        if not os.path.exists(self.record_path):
            os.mkdir(self.record_path)
        self.im_size = cfg.image_size
        self.classes = cfg.classes
        self.cell_size = cfg.cell_size
        self.box_per_cell = cfg.box_per_cell
        self.num_class = cfg.num_class
        self.data_path  = cfg.data_path
        
        self.class2ind = dict(zip(self.classes,range(len(self.classes))))
        
        
        
        
        if phase=='train':
            self.txt_path = os.path.join(cfg.data_path, 'ImageSets', 'Main', 'trainval.txt')
            self.train_record_path = os.path.join(self.record_path,'train.record')
            self.input_record_path = self.train_record_path
            if not os.path.exists(self.train_record_path):
                self.make_traindata_set(self.txt_path,self.train_record_path)
        
        
    def read_im(self,im_path):
        im = cv2.imread(im_path)
        h,w = im.shape[0],im.shape[1]
        im = cv2.resize(im,(self.im_size,self.im_size))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)
        im = im/256.0
        return im,w,h
    
    
    def read_train_xml(self,xml_path,w,h):
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        labels = np.zeros([self.cell_size,self.cell_size,self.box_per_cell,self.num_class+5],dtype=np.float32)
#        w_ratio = 1.0*self.im_size/w
#        h_ratio = 1.0*self.im_size/h
        
        for obj in objs:
            box = obj.find('bndbox')
            x_min = float(box.find('xmin').text)/w
            y_min = float(box.find('ymin').text)/h
            x_max = float(box.find('xmax').text)/w
            y_max = float(box.find('ymax').text)/h
            cls_ind = self.class2ind[obj.find('name').text]
            boxes = [(x_max+x_min)/2.0,(y_max+y_min)/2.0,x_max-x_min,y_max-y_min]
            
            x_ind = int(boxes[0]*self.cell_size)
            y_ind = int(boxes[1]*self.cell_size)
                 
#            boxes = self.encode_label(boxes)
            
            
            labels[y_ind,x_ind,:,0] = 1
            labels[y_ind,x_ind,:,1:5] = boxes
            labels[y_ind,x_ind,:,5+cls_ind]  = 1
            # debug
            if(0):
                if x_min>w or x_max>w or x_min<0 or x_max<0 or x_min>x_max:
                    print('x is wrong')
                if y_min>h or y_max>h or y_min<0 or y_max<0 or y_min>y_max:
                    print('y is wrong')            
        
        return labels
#    def encode_label(self,boxes):
#        '''
#        input:
#        boxes:[x,y,w,h]~(0,1)
#        
#        out_put:
#        encode_boxes:[5,4]->[num_anchors,4]
#        '''
#        tx = boxes[0]*self.cell_size-int(boxes[0]*self.cell_size)
#        ty = boxes[1]*self.cell_size-int(boxes[1]*self.cell_size)
#        N = len(cfg.anchors)
#        anchors = cfg.anchors
#        encode_boxes = []
#        for i in range(N):
#            tw = np.log(boxes[3]*self.cell_size/anchors[i,0])
#            th = np.log(boxes[3]*self.cell_size/anchors[i,1])
#            encode_boxes.append([tx,ty,tw,th])
#        return encode_boxes
    
    def make_traindata_set(self,txt_path,out_path):
        
        # 初始化writer
        writer = tf.python_io.TFRecordWriter(out_path)
        # 读样本名
        with open(txt_path, 'r') as f:
            data_name = [x.strip() for x in f.readlines()]
            
        for i,data_i in enumerate(data_name):
            # 图像和xml的路径
            im_path = os.path.join(self.data_path,'JPEGImages',data_i+'.jpg')
            label_path = os.path.join(self.data_path,'Annotations',data_i+'.xml')
            # 读取图像和xml文件
            im,w,h = self.read_im(im_path)
            label = self.read_train_xml(label_path,w,h)
            # 定义tf example
            tf_example = self._make_train_tfexample(im,label)
            writer.write(tf_example.SerializeToString())
            if i%200==0:
                print('%d/%d'%(i, len(data_name)))
        writer.close()
        
    def build_data(self):
        if self.phase == 'train':
            return self.build_train_data()
        
        
    def build_train_data(self):
        dataset = tf.data.TFRecordDataset(self.input_record_path,num_parallel_reads = 16)
        dataset = dataset.map(self._parse_train_function,num_parallel_calls=16)
#        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)
        iterator = dataset.make_initializable_iterator()
        images,labels= iterator.get_next()
        
        
        
        return iterator,images,labels
    
    
    def _parse_train_function(self,example_proto):
        

        def random_adjust_image(image):
            
            oder = np.random.choice([0,0,1,2,3,4],[4],replace=False)
            
            for i in oder:
                if i==0:
                    pass
                if i==1:
                    image = tf.image.random_brightness(image,max_delta=16./255.)
                if i==2:
                    image = tf.image.random_saturation(image,lower=0.8,upper=1.2)
                if i==3:
                    image = tf.image.random_hue(image,max_delta=0.2)
                if i==4:
                    image = tf.image.random_contrast(image,lower=0.8,upper=1.2)
            return tf.clip_by_value(image,0.0,1.0)
        
        
        
        
        features = {
                'image': tf.FixedLenFeature((), tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.string, default_value="")
                }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['image'], tf.float32)
        image = tf.reshape(image, [self.im_size, self.im_size, 3])
        # data augmentation
        image = random_adjust_image(image)
        
        label = tf.decode_raw(parsed_features['label'], tf.float32)
        label = tf.reshape(label,[self.cell_size,self.cell_size,self.box_per_cell,self.num_class+5])

        return image,label    
    
    def _make_train_tfexample(self,im,label):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': self._bytes_feature(tf.compat.as_bytes(im.tostring())),
            'label': self._bytes_feature(tf.compat.as_bytes(label.tostring()))
    		}))
        return tf_example
    
    
    
    
    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



if __name__ == '__main__':    
    data = pascal_voc('train')


    iterator,images,labes = data.build_data()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        im,la = sess.run((images,labes))


