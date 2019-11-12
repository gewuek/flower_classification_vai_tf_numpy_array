#! /usr/bin/python3
# coding=utf-8
#####################

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# May meet problem when loading a new pb file
tf.contrib.resampler

# Set the class number, input/output node names
CLASS_NUM = 5
# INPUT_NODE = 'conv2d_input_2' # Get from "Freeze the model" flow
# OUTPUT_NODE = 'dense_1_2/Softmax' # Get from "Freeze the model" flow
f = open("./freeze_input_output_node_name.txt", "r+")
curline = f.readline()
INPUT_NODE = curline.strip()
curline = f.readline()
OUTPUT_NODE = curline.strip()
f.close()

# Load images and Normalization
X = np.load("features.npy") / 255.0
# Load labels and change y label to one hot
y = np.load("label.npy")
y_one_hot = np.squeeze(np.eye(CLASS_NUM)[np.array(y).reshape(-1)])

# Function to load pb file as a graph
def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

# Load quantized pb file
graph = load_pb('./quantize_results/quantize_eval_model.pb')

# Name the input/output node
input_node = graph.get_tensor_by_name(INPUT_NODE + ':0')
output_node = graph.get_tensor_by_name(OUTPUT_NODE + ':0')
print(input_node.shape)
print(output_node.shape)

# Evalate the graph
with tf.Session(graph=graph) as sess:	
    predict = sess.run(output_node, feed_dict={input_node: X[:10]})
print(predict)
print(y[:10])
loss = tf.losses.softmax_cross_entropy(y_one_hot[:10], predict)
sess_loss = tf.Session()
loss = sess_loss.run(loss)
print(loss)
