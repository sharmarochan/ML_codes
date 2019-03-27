# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 06:00:20 2019

@author: tensor19
"""

import keras
import PIL
#import cv2
import numpy as np

import os, glob



from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array



base_dir = r'E:\ML_training\waffle-pancake-images\waffle_pancakes\train\waffles'


'''
os.chdir(base_dir)

waffle_train_img_name = glob.glob("*.png")


print(len(waffle_train_img_name))




base_dir = r'E:\ML_training\waffle-pancake-images\waffle_pancakes\train\pancakes'

os.chdir(base_dir)

pancake_train_img_name = glob.glob("*.png")


print(len(pancake_train_img_name))





base_dir = r'E:\ML_training\waffle-pancake-images\waffle_pancakes\test\waffles'

os.chdir(base_dir)

waffle_test_img_name = glob.glob("*.png")


print(len(waffle_test_img_name))




base_dir = r'E:\ML_training\waffle-pancake-images\waffle_pancakes\test\pancakes'

os.chdir(base_dir)

pancake_test_img_name = glob.glob("*.png")


print(len(pancake_test_img_name))

'''


def get_class_names(base_dir):
    os.chdir(base_dir)
    return (glob.glob("*.png"))


get_class_names(base_dir)


test_waffel = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\test\waffles"
train_waffel  = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\train\waffles"


train_pancakes = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\train\pancakes"
test_pancakes = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\test\pancakes"

waffle_test_img_name = get_class_names(test_waffel)
waffle_train_img_name = get_class_names(train_waffel)


pancake_test_img_name = get_class_names(test_pancakes)
pancake_train_img_name = get_class_names(train_pancakes)




waffle_train = []
dir = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\train\waffles"
os.chdir(dir)
import numpy as np
for name in waffle_train_img_name:
    im = load_img(name, target_size=(224,224,3))
    im=img_to_array(im)
    im = np.expand_dims(im,axis = 0)
    im = preprocess_input(im)
    waffle_train.append(im[0])
    
    
    

    
waffle_test = []
dir = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\test\waffles"
os.chdir(dir)
import numpy as np
for name in waffle_test_img_name:
    im = load_img(name, target_size=(224,224,3))
    im=img_to_array(im)
    im = np.expand_dims(im,axis = 0)
    im = preprocess_input(im)
    waffle_test.append(im[0])
    
    


pancakes_test = []
dir = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\test\pancakes"
os.chdir(dir)
import numpy as np
for name in pancake_test_img_name:
    im = load_img(name, target_size=(224,224,3))
    im=img_to_array(im)
    im = np.expand_dims(im,axis = 0)
    im = preprocess_input(im)
    pancakes_test.append(im[0])




pancakes_train = []
dir = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\train\pancakes"
os.chdir(dir)
import numpy as np
for name in pancake_train_img_name:
    im = load_img(name, target_size=(224,224,3))
    im=img_to_array(im)
    im = np.expand_dims(im,axis = 0)
    im = preprocess_input(im)
    pancakes_train.append(im[0])
    
    

type(waffle_train)
waffle_train = np.array(waffle_train)
waffle_train.shape

type(pancakes_train)
pancakes_train = np.array(pancakes_train)
pancakes_train.shape


train = np.concatenate([waffle_train,pancakes_train],axis=0)

train.shape

y_train = [1]*572+[0]*375
y_train = np.array(y_train)
y_train.shape

model = Xception(include_top=False, weights="imagenet",pooling="avg", input_shape=(224,224,3))


#pooling = 'avg' flatten everything
model.summary()

import time
start_time = time.time()

X_features = model.predict(train)

print("--- %s seconds ---" % (time.time() - start_time))


X_features.shape   #(947, 2048) 947 is no of images in the train, 2048 is due to avg pooling






import sklearn.linear_model as linear_model


clf = linear_model.LogisticRegression()

start_time = time.time()


clf.fit(X_features, y_train)


print("--- %s seconds ---" % (time.time() - start_time))


#data generators


from keras.preprocessing.image import ImageDataGenerator


generator = ImageDataGenerator(horizontal_flip=True)  #prepocessing function is present in the ImageDataGenerator


save_dir = r"E:\ML_training\aug_images"
generator_path = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\train"
generator_train = generator.flow_from_directory(generator_path, batch_size=32, save_to_dir=save_dir)


next(generator_train)  #like as for i in range 1


########################################################################





from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.preprocessing.image import ImageDataGenerator




def preprocess(im):
    im = img_to_array(im)
    im =np.expand_dims(im,axis=0)
    im = preprocess_input(im)
    return im[0]


path = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\train"

gen = ImageDataGenerator(rotation_range=40,vertical_flip=True,preprocessing_function = preprocess)

train_gen = gen.flow_from_directory(path,target_size=(150,150), batch_size=64)


##test

path_test = r"E:\ML_training\waffle-pancake-images\waffle_pancakes\test"
test_gen = gen.flow_from_directory(path_test,target_size=(150,150), batch_size=64, shuffle=False)   #shuffel should be off in test

base_model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

from keras.layers import Dense
from keras.models import Model

x = base_model.output
x = Dense(units=1024,activation="relu")(x)


x = Dense(units=2,activation="softmax")(x)

model = Model(base_model.input,x)


model.summary()


for layer in base_model.layers:
    layer.trainable = False


model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])
model.fit_generator(train_gen, validation_data = test_gen, steps_per_epoch=947//64, validation_steps=406//64, epochs=2)









