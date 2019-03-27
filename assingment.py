# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:10:07 2019

@author: tensor19
"""

import os

from matplotlib.image import imread


root = r"E:\ML_training\final_data"
train_arr = []

for path, subdirs, files in os.walk(root):
    for name in files:
        full_path = os.path.join(path, name)
        train_arr.append(full_path)
        
        name_upto_folder =  os.path.dirname(full_path)        

print(len(train_arr))
        
from skimage.transform import rescale, resize, downscale_local_mean

import pylab as pl
import numpy as np


import csv
 




from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.preprocessing.image import load_img,img_to_array

for imagePath in train_arr:
    img = load_img(imagePath,target_size=(224,224))
    
    image = np.asarray(img)
    image = np.expand_dims(image,axis=0)
    model = VGG16(weights='imagenet')
    proba = model.predict(image)
    
    
    
    output = decode_predictions(proba,top=1)[0]
    for i in output:
        with open("output.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(i)
            
            
            
            
    break







model = VGG16(weights="imagenet")

label =[]
n=[]



for imagePath in train_arr:
    im = load_img(imagePath,target_size=(224,224,3))
    im = img_to_array(im)
    im = np.expand_dims(im,axis=0)
    im = preprocess_input(im)
    pred = model.predict(im)
    l = decode_predictions(pred,top=1)[0][0][1]
    label.append(l)
#    print(l)
    n.append(name)
    

import pandas as pd
table = pd.DataFrame({'name':n,'label':label})
table.head()


