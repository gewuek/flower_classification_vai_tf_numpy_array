#! /usr/bin/python3
# coding=utf-8
#####################


import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# Load images and labels, do the normalization
X = np.load("features.npy") / 255.0
y = np.load("label.npy")

# Load the model and check the model summary
loaded_model = keras.models.load_model("./flower_classification_weights.h5")
loaded_model.summary()

# Get the loss and accurary
loss,acc = loaded_model.evaluate(X, y)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))