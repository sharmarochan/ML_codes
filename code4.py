
import keras
import PIL
#import cv2
import numpy as np


from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.preprocessing.image import load_img,img_to_array



path = r"E:\temp\00000.png"


im=load_img(path,target_size=(224,224))


im=img_to_array(im)

im.shape

im=preprocess_input(im)



im=np.expand_dims(im,axis=0)

im.shape

model=VGG16(weights='imagenet')

predictions = model.predict(im)


decode_predictions(predictions,top=1)

decode_predictions(predictions)  #top 5 predictions


model.summary()

#build owr own model using a pre-trained model

model_no_top = VGG16(include_top=False, weights='imagenet')


model_no_top.summary()

predictions = model_no_top.predict(im)

predictions.shape


predictions[:,:,:,0].reshape((7,7))


import matplotlib.pyplot as plt


plt.imshow(predictions[:,:,:,0].reshape((7,7)),cmap="gray")   #visuallise the feature image

#decode_predictions(predictions,top=1)


from keras.models import Model

model_no_top.summary()

mod = Model(model_no_top.input , model_no_top.get_layer("block1_conv1").output )

mod.summary()



pred_smaller = mod.predict(im)

pred_smaller.shape

plt.imshow(pred_smaller[:,:,:,10].reshape((224,224)),cmap="gray")


#############################
mod2 = Model(model_no_top.input , model_no_top.get_layer("block2_conv2").output )

mod2.summary()



pred_smaller2 = mod2.predict(im)

pred_smaller2.shape  #to get the height and width of the output image

plt.imshow(pred_smaller2[:,:,:,10].reshape((112,112)),cmap="gray")







