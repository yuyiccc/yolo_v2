# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:07:33 2018

@author: Administrator
"""

import os  
from tensorflow.python import pywrap_tensorflow  
model_dir="G:\\tensorflow\\task4\\output\\init_weight\\yolo_weights.ckpt" #checkpoint的文件位置  
# Read data from checkpoint file  
reader = pywrap_tensorflow.NewCheckpointReader(model_dir)  
var_to_shape_map = reader.get_variable_to_shape_map()  
# Print tensor name and values  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  #输出变量名  
    print(reader.get_tensor(key))   #输出变量值  