#!/bin/bash

# top.hwh is from ZCU102 DPU TRD
dlet -f top.hwh

dcf_file_name=$(find -name *.dcf)

# DNNC command to compile pb file into elf file
dnnc --parser=tensorflow \
    --frozen_pb=quantize_results/deploy_model.pb \
    --dcf=$dcf_file_name \
    --cpu_arch=arm64 \
    --output_dir=flower_classification \
    --save_kernel \
    --mode normal \
    --net_name=flower_classification

# remove the .dcf file
rm $dcf_file_name
