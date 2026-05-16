#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export foldname=$1
export weight_file=$2
export MONAI_DATA_DIRECTORY="/redactedsharedrive/pteng/pteng-public/dataset/stp/unetr_results/${foldname}"

python train.py dataset_${foldname}.json --pretrain True --pretrained_weights $weight_file
