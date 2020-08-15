#!/bin/bash

INPUT_NODE=$(sed '1q;d' ./freeze_input_output_node_name.txt)
OUTPUT_NODE=$(sed '2q;d' ./freeze_input_output_node_name.txt)

vai_q_tensorflow quantize \
    --input_frozen_graph ./frozen_graph.pb \
    --input_nodes $INPUT_NODE \
    --input_shapes ?,128,128,3 \
    --output_nodes $OUTPUT_NODE \
    --input_fn flower_classification_input_fn.calib_input \
    --method 1 \
    --gpu 0 \
    --calib_iter 10 \
    --output_dir ./quantize_results
