

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
## for gpu computation.. tensorflow 2.0 bug
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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
  #schotastic gradient descent sgd
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
  #  opt = SGD(lr=0.01)
  
    model.compile(loss = "mse", optimizer = sgd ,metrics=['accuracy'])

 
    #model.compile(loss = "categorical_crossentropy", optimizer = "adam",metrics=['accuracy'])
 
 
    if weights_path:
        model.load_weights(weights_path)
 
    return model
path = 'benzema_weigths_original_256_96.06_banglalekha.h5'
model = benzema_92(weights_path=path)
model.summary()

model.load_weights('benzema_weigths_original_256_96.06_banglalekha.h5')
model.save_weights('benzema_weigths_original_256_96.06_banglalekha.h5')

#path = 'audri_benzema_transferred.h5'

#augmentation 
train_datagen = ImageDataGenerator(
  rotation_range=35, 
  width_shift_range=.2,
   height_shift_range=.2,
    brightness_range=None,
     shear_range=.2, 
     zoom_range=.2, 
    horizontal_flip=True, 
     vertical_flip=False,
     #normalization
     rescale = 1/255.,
     validation_split = .2,

      
dtype='float32')


train_generator = train_datagen.flow_from_directory(
    directory='Images',
    target_size= (64,64),
    color_mode = 'grayscale',
    batch_size = 64,
    class_mode= 'categorical',
    subset= 'training')
valid_generator =  train_datagen.flow_from_directory(
    directory= 'Images',
    color_mode = 'grayscale',
    target_size=(64,64),
    batch_size= 64,
    class_mode = 'categorical',
    subset= 'validation'
)

# because we are using interator we need to use fit generator
history = model.fit_generator(
    train_generator,
    validation_data = valid_generator,
    epochs =3,
    callbacks=[
            ModelCheckpoint(filepath=path)
        ]
    )

model.save_weights('benzema_weigths_original_256_96.06_banglalekha.h5')
