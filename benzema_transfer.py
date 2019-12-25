import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2 as c 
print(tf.__version__)

#keras = tf.keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,Dropout
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
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

model = load_model('benzema_sgd_best_512_96.33.h5')
model.summary()

#for layer in model.layers:
 #   layer.trainable = False

tf_model = Sequential()
tf_model.add(model)
tf_model.add(Flatten())
tf_model.add(Dense(254, activation = 'relu'))
#tf_model.add(Dropout(0.2))
#tf_model.add(Dense(1024, activation = 'relu'))
#tf_model.add(Dropout(0.2))
tf_model.add(Dense(254, activation = 'relu'))
tf_model.add(Dropout(0.2))
tf_model.add(Dense(128, activation = 'relu'))
tf_model.add(Dropout(0.2))
#tf_model.add(Dense(256, activation = 'relu'))
tf_model.add(Dense(10, activation = 'softmax'))
tf_model.compile(optimizer=Adam(0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tf_model.summary()
path = 'audri_benzema_transferred.h5'


train_datagen = ImageDataGenerator(
  rotation_range=35, 
  width_shift_range=.2,
   height_shift_range=.2,
    brightness_range=None,
     shear_range=.2, 
     zoom_range=.2, 
    horizontal_flip=True, 
     vertical_flip=False,
     rescale = 1/255.,

      
dtype='float32')


train_generator = train_datagen.flow_from_directory(
    directory='Images',
    target_size= (64,64),
    color_mode = 'grayscale',
    batch_size = 64,
    class_mode= 'categorical')


history = tf_model.fit_generator(
    train_generator,
    epochs = 50,
    callbacks=[
            ModelCheckpoint(filepath=path)
        ]
    )


'''path = 'normal_abnormal_benzema_transferred_without_aug.h5'
model_log = tf_model.fit(X_train,y_train,
batch_size= 8,
epochs=20,
verbose= 1,
validation_data = (X_test,y_test),
shuffle = True,
callbacks= [
    ModelCheckpoint(filepath = path)
])

train_datagen = ImageDataGenerator(
  rotation_range=35, 
  width_shift_range=.2,
   height_shift_range=.2,
    brightness_range=None,
     shear_range=.2, 
     zoom_range=.2, 
     horizontal_flip=True, 
     vertical_flip=False,
      
dtype='float32')

valid_datagen = ImageDataGenerator(
    rescale = 1/255.0
)

path = 'normal_abnormal_benzema_transferred.h5'


training_generator = train_datagen.flow(X_train, y_train, batch_size=8)
validation_generator = valid_datagen.flow(X_test,y_test,batch_size= 8)

X_test= X_test/255.0
history = tf_model.fit_generator(
    training_generator,
    epochs = 50,
    validation_data= (X_test,y_test),
    callbacks=[
            ModelCheckpoint(filepath=path)
        ]
    )
'''