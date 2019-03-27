# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 04:45:03 2019

@author: tensor19
"""

import keras
import os
import pandas as pd


train = pd.read_csv("fashion-mnist_train.csv") 
test = pd.read_csv("fashion-mnist_test.csv") 

train.head()


#separate pixels and labels

y_train = train['label'].values
y_test = test['label'].values

x_train = train.drop("label",axis=1).values
x_test = test.drop("label",axis=1).values

y_test

#create a label matrix


y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)


#assemble model

from keras.layers import Dense, Activation

from keras.models import Sequential

model = Sequential()



model.add(Dense(units = 32, activation = 'relu', input_shape=(784,)))   #input shape is no of columns

model.add(Dense(units =64, activation = "relu")) 

model.add(Dense(units = 10, activation="softmax")) 
model.summary()


model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32, epochs=10,validation_split = 0.10)
#train model












"""
flattening of the images is not good, it will distroy the image spatial imformation

output = (width+depth-2padding)/stride + 1

"""


