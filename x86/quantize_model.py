f = open("./freeze_input_output_node_name.txt", "r+")
curline = f.readline()
INPUT_NODE = curline.strip()
curline = f.readline()
OUTPUT_NODE = curline.strip()
f.close()

!decent_q quantize \
	--input_frozen_graph ./frozen_graph.pb \
	--input_nodes $INPUT_NODE \
	--input_shapes ?,128,128,3 \
	--output_nodes $OUTPUT_NODE \
	--input_fn flower_classfication_input_fn.calib_input \
	--method 1 \
	--gpu 0 \
	--calib_iter 10 \
	--output_dir ./quantize_results