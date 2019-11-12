#! /usr/bin/python3
# coding=utf-8
#####################


import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = './flowers/'
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = 128
training_data = []

# Function to go through the category folders
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        label_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resized_array, label_num])
            except Exception as e:
                print("Exception here") #pass

create_training_data()

# Print out the picture numbers
print(len(training_data))

# Random shuffle the dataset
random.shuffle(training_data)

# Define the features(images) and labels
X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)

# Check the X dataset if necessary
#from PIL import Image
#img = Image.fromarray(X[0])
#img.show()

# Store data into NPY files
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
np.save("features.npy", X)
np.save("label.npy", y)

