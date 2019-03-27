# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:12:13 2019

@author: tensor19
"""



"""
flattening of the images is not good, it will distroy the image spatial imformation

output = (width+depth-2padding)/stride + 1




we must have as many images as the number of the parameters in summary of the model
"""

#def conv_cal(o,w,k,p,s):
#    o = ((w-k+(2*p))/s)+1
#    print(o)
#    
#    
#conv_cal(28,32,5,0,1)  


def conv_cal2(input,output,padding,stride):
    kernel = input+(2*padding)+stride*(1-output)
    print(kernel)
    
conv_cal2(input=32,output=28,padding=0,stride = 1)





from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.layers import Dense, Activation

from keras.models import Sequential

model_LeeNet = Sequential()

model_LeeNet.add(Conv2D(filters=6,kernel_size=(5,5), strides=(1,1), padding ="valid", input_shape=(32,32,1)))
model_LeeNet.add(Conv2D(filters=6,kernel_size=(5,5), strides=(1,1), padding ="valid", input_shape=(30,30,1)))


model_LeeNet.summary()