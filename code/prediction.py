# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 22:56:06 2018

@author: yuyi
"""

import tensorflow as tf
import cv2


from yolo_v2 import yolo_v2
import config as cfg
import utili as u

image_path = '/home/yuyi/Desktop/detect/data/test_img/3.jpg'
model_path = '/home/yuyi/Desktop/detect/output/model_weight/yolo_v2-30000'



def main():
    
    image_resize,image_orig,h,w = u.read_im(image_path)
    im = tf.placeholder(tf.float32,shape=[1,cfg.image_size,cfg.image_size,3])
    yolo_model = yolo_v2(im,is_trainning=False)
    obj_probs,class_probs,bboxes_probs = yolo_model.predict
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,model_path)
        obj_value,class_value,bboxes_value = sess.run((obj_probs,class_probs,bboxes_probs),feed_dict={im:image_resize})
    result = u.postprosessing(obj_value,class_value,bboxes_value,w,h)
    img_detection = u.draw_box(image_orig,result)
    
    cv2.imwrite("detection.jpg", img_detection)
    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)
    tf.reset_default_graph()
    
    
if __name__=='__main__':
    main()