#!/bin/bash

TARGET=ZCU102
NET_NAME=flower_classification
ARCH=${CONDA_PREFIX}/arch/dpuv2/${TARGET}/${TARGET}.json

# DNNC command to compile pb file into elf file
vai_c_tensorflow \
    --frozen_pb quantize_results/deploy_model.pb \
    --arch ${ARCH} \
    --output_dir flower_classification \
    --net_name ${NET_NAME} \
    --options "{'save_kernel':''}"


