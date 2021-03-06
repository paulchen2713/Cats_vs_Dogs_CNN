# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 14:33:09 2021

@author: Paul
@file: Cats_vs_Dogs_CNN.py

@dependencies: 
    conda env: tf2
    tf.__version__: 2.1.0
    tf.keras.__version__: 2.2.4
    keras.__version__: 2.3.1
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from tensorflow.keras import losses

from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
(noticed: the following codes activate only when using GPU to run the model, 
 but I'm not using GPU support I think. )

# check to be sure that TensorFlow is able to identify the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# set_memory_growth() allocate only as much GPU memory as needed at a given time, 
# and continues to allocate more when needed
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(IndexError                                Traceback (most recent call last)
<ipython-input-3-df3199446273> in <module>
     37 # set_memory_growth() allocate only as much GPU memory as needed at a given time,
     38 # and continues to allocate more when needed
---> 39 tf.config.experimental.set_memory_growth(physical_devices[0], True)

IndexError: list index out of range)
"""

# I. data preparation
# organize the kaggle dogs vs cats train data into train, validation, and test directories

# SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: 
# truncated \UXXXXXXXX escape(change "\" into "/" in the file path)
os.chdir('D:/Deep_Learning_Projects/CNN_Cats_vs_Dogs/dogs_vs_cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
    
    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')      
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')        
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')      
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../../')

# assign the path to path variables for upcoming preprocessing
train_path = 'D:/Deep_Learning_Projects/CNN_Cats_vs_Dogs/dogs_vs_cats/train'
valid_path = 'D:/Deep_Learning_Projects/CNN_Cats_vs_Dogs/dogs_vs_cats/valid'
test_path = 'D:/Deep_Learning_Projects/CNN_Cats_vs_Dogs/dogs_vs_cats/test'

# use Keras' ImageDataGenerator class to create batches of data from the train, valid, and test directories.
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

"""
# verify the number of images and classes are correct
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
# Fix for Keras 2.1: num_class is now num_classes for DirectoryIterator
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2
"""

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(imgs)
# print(labels)

# II. Build a Convolutional Neural Network
"""
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(units=2, activation='softmax'))
"""

vgg16_model = tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()
"""
parameters = count_params(vgg16_model)
assert parameters['Non-trainable params'] == 0
assert parameters['Trainable params'] == 138357544
"""
# print(type(vgg16_model))
# <class 'tensorflow.python.keras.engine.training.Model'>

model = tf.keras.Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()
"""
parameters = count_params(model)
assert parameters['Non-trainable params'] == 0
assert parameters['Trainable params'] == 134260544
"""
for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))

# call summary() to get a quick visualization
model.summary()
"""
parameters = count_params(model)
assert parameters['Non-trainable params'] == 134260544
assert parameters['Trainable params'] == 8194
"""

# III. Training
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

"""
model.fit_generator(train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose = 2
)
"""
model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=5,
    verbose=2
)

# IV. Prediction
# inference process - the model make predictions for the test_samples based on 
# what it's learned from the train_samples, this process called "inference."

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

# print(test_batches.classes)

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

# look only at the most probable prediction
# print(np.round(predictions))

# V. Confusion Matrix
# create the confusion matrix by calling plot_confusion_matrix() function from 
# scikit-learn and assign it to the variable cm(confusion matrix)
# pass the true labels(test_labels) and prediction labels(round_predictions) to cm(confusion matrix)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

# the function plot_confusion_matrix() came directly from scikit-learn???s website
# this is code that they provide in order to plot the confusion matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# print(test_batches.class_indices)
# {'cat': 0, 'dog': 1}

# define the labels for the confusion matrix, which is "cat" or "dog."
cm_plot_labels = ['cat','dog']
# plot the confusion matrix by using the plot_confusion_matrix() function
# pass in the confusion matrix cm and the labels cm_plot_labels, and a title Confusion Matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
                           

