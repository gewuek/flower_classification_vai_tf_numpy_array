#! /usr/bin/python3
# coding=utf-8
#####################


import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# Load images and labels from npy files
X = np.load("features.npy")
y = np.load("label.npy")

# Normalization
X = X / 255.0

# Define the network model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), padding="same", activation=tf.nn.relu, input_shape=(128, 128, 3)),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Check the summary, not necessary
model.summary()

# Training the model
model.fit(X, y, epochs=10)

# Save the model to H5 file
model.save("flower_classification_weights.h5")