#! /usr/bin/python3
# coding=utf-8
#####################


import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# Define the freeze function, outptut freezed graph
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
    return frozen_graph
    
# Load trained model, set learning phase to 0
keras.backend.set_learning_phase(0)
loaded_model= keras.models.load_model('./flower_classification_weights.h5')

# make list of output and input node names
input_names=[out.op.name for out in loaded_model.inputs]
output_names=[out.op.name for out in loaded_model.outputs]
print('input  node is{}'.format(input_names))
print('output node is{}'.format(output_names))

f = open("freeze_input_output_node_name.txt", "w+")
# f.write('{}'.format(input_names[input_names.find("'")+1:input_names.find("'")]) + "\n")
f.write('{}'.format(input_names[0]) + "\n")
f.write('{}'.format(output_names[0]) + "\n")
f.close()

# Freeze graph
frozen_graph = freeze_session(keras.backend.get_session(), output_names=output_names)
# Store graph to pb(Protocol Buffers) file
tf.train.write_graph(frozen_graph, "./", "frozen_graph.pb", as_text=False)
