image_dir = 'eee.png'



import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2 as c 
print(tf.__version__)

#keras = tf.keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,Dropout,MaxPooling2D,Conv2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class_name = ['o','aa','e','ee','u','uu','ri','eee','oi','o','ou']

def benzema_92(weights_path=None):
    model = Sequential()
   
    model.add(Conv2D(32, (5, 5), input_shape=(64,64,1), padding='same',activation='relu'))
  #  model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),  padding='same',activation='relu'))
   # model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))
   
 
    model.add(Conv2D(64, (3, 3),  padding='same',activation='relu'))
   # model.add(Conv2D(64, (3, 3),  padding='same',activation='relu'))
    model.add(Conv2D(64, (3, 3),  padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #output size = 32*32*64

    model.add(Conv2D(128, (3,3),  padding='same',activation='relu'))
    model.add(Conv2D(128, (3,3),  padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #output size = 16*16*128
   
    model.add(Conv2D(256, (3,3),  padding='same',activation='relu'))
    model.add(Conv2D(256, (3,3),  padding='same',activation='relu'))
    model.add(Conv2D(256, (3,3),  padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
   
 
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
   # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
  #  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  #  opt = SGD(lr=0.01)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
  #  opt = SGD(lr=0.01)
  
    model.compile(loss = "mse", optimizer = sgd ,metrics=['accuracy'])

 
    #model.compile(loss = "categorical_crossentropy", optimizer = "adam",metrics=['accuracy'])
 
 
    if weights_path:
        model.load_weights(weights_path)
 
    return model

model = benzema_92()
model.load_weights('benzema_weigths_original_256_96.06_banglalekha.h5')
model.summary()

image = c.imread(image_dir)
image_processed  = c.cvtColor(image, c.COLOR_RGB2GRAY)
image_processed = c.resize(image_processed,interpolation=c.INTER_CUBIC,dsize= (64,64))

print(image_processed.shape)

image_processed = image_processed/255.0

image_processed = np.expand_dims(image_processed, axis= 0)
image_processed = np.expand_dims(image_processed, axis= 3)
print(image_processed.shape)

y_p = model.predict_classes(image_processed)
print(class_name[y_p[0]-1])

c.imshow(class_name[y_p[0]-1], image)
c.waitKey()

