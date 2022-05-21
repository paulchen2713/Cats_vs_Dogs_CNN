# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:45:51 2020

@author: Paul
@file: Cats_vs_Dogs_CNN.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
# normally tensorflow.keras should be the right command, but it doesn't work, don't know why...
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
"""

# I. data preparation
# organize the kaggle dogs vs cats train data into train, validation, and test directories

# SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: 
# truncated \UXXXXXXXX escape(change "\" into "/" in the file path)
os.chdir('D:/computer science lab/dogs_vs_cats')
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
train_path = 'D:/computer science lab/dogs_vs_cats/train'
valid_path = 'D:/computer science lab/dogs_vs_cats/valid'
test_path = 'D:/computer science lab/dogs_vs_cats/test'

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

# "AssertionError" means the assertion statement is False
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

plotImages(imgs)
print(labels)


