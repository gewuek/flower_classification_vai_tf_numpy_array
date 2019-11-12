#import cv2
import numpy as np
#import os
#import tensorflow as tf
#from tensorflow import keras


def calib_input(iter):
	X = np.load("features.npy") / 255.0
	images = X[:100]
	return {"conv2d_input": images}

