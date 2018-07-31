# -*- coding: utf-8 -*-
"""
Created on Thu May 31 23:24:12 2018

@author: yuyi
"""

import tensorflow as tf
import os
import re
#from tensorflow.python import debug as tf_debug

import config as cfg
from yolo_v2 import yolo_v2
from preprosess_data import pascal_voc


class Train():
    def __init__(self,yolo_v2,iterations):
        self.iterations = iterations
        self.yolo = yolo_v2
        
        self.saver_iter = cfg.saver_iter
        self.summary_iter  = cfg.summary_iter
        self.initial_learn_rate = cfg.learning_rate
        
        self.summary_path = os.path.join(cfg.out_path,'summary')
        self.save_weight_path = os.path.join(cfg.out_path,'model_weight','yolo_v2')
        self.variable = tf.trainable_variables()
        
        self.variable_restore = [v for v in self.variable 
                                 if(re.split('[:,/,_]',v.name)[0] not in cfg.exclude_variable
                                 and v.name not in cfg.exclude_variable)]
        self.restore_saver = tf.train.Saver(self.variable_restore)

        
        self.train_saver = tf.train.Saver()
        
        self.global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate,
                                                     self.global_step,5000,
                                                     decay_rate=0.5,
                                                     staircase=True,
                                                     name='learning_rate')
        
        # summary learning rate
        tf.summary.scalar('learning rate',self.learn_rate)
        
        self.optimizer = tf.train.MomentumOptimizer(self.learn_rate,0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimiz_op = self.optimizer.minimize(self.yolo.total_loss,global_step=self.global_step)
        
        # summary
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.summary_path)
        
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
#        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(tf.global_variables_initializer())
        
        print('====用预训练初始化权重=====')
        self.restore_saver.restore(self.sess,cfg.ini_weight_path)
        
        
    def train(self):
        self.sess.run(self.iterations.initializer)
        i = 0
        while True:
            try:
                _,summary,loss = self.sess.run((self.optimiz_op,self.summary_op,self.yolo.total_loss))
                if i%self.summary_iter==0:
#                if True:
                    self.writer.add_summary(summary,i)
                    print('====step:%d-loss:%.4f'%(i,loss))
                if i%self.saver_iter==0:
                    self.train_saver.save(self.sess,self.save_weight_path,i)
            except tf.errors.OutOfRangeError:
                break
            i += 1
        self.writer.close()
        
        
        
def main():
        data = pascal_voc('train')
        iterations,images,labels = data.build_data()
        yolo = yolo_v2(images,labels)
        train = Train(yolo,iterations)
        print('=====开始训练=====')
        train.train()
        
if __name__=='__main__':
    main()
        
