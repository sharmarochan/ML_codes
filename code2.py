# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:23:03 2019

@author: tensor19

CNN simple
"""

"

import keras
import os
import pandas as pd


train = pd.read_csv("fashion-mnist_train.csv") 
test = pd.read_csv("fashion-mnist_test.csv") 


y_train = train['label'].values
y_test = test['label'].values

x_train = train.drop("label",axis=1).values
x_test = test.drop("label",axis=1).values

y_train.shape
x_train = x_train.reshape((60000,28,28,1))



y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)

x_train[0,:,:,:].shape

x_train.shape
print(type(train))


x_test = x_test.reshape((10000,28,28,1))

x_test.shape


import matplotlib.pyplot as plt
plt.imshow(x_test[22,:,:,:].reshape((28,28)),cmap="gray")


from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.layers import Dense, Activation

from keras.models import Sequential

model_cnn = Sequential()

model_cnn.add(Conv2D(filters=6,kernel_size=(5,5),input_shape=(28,28,1)))

model_cnn.add(MaxPool2D())

model_cnn.add(Conv2D(filters=10,kernel_size=(3,3)))

model_cnn.add(Flatten())




model_cnn.add(Dense(units=40,activation="relu"))
model_cnn.add(Dense(units=10,activation="softmax"))


model_cnn.summary()


model_cnn.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model_cnn.fit(x_train,y_train,batch_size=32, epochs=10,validation_split = 0.10)




