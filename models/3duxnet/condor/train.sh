#!/bin/bash
export foldname=$1
export weight_file=$2

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RESULTS_DIRECTORY="/redactedsharedrive/pteng/pteng-public/dataset/stp/3duxnet_results/${foldname}"
export JSON_FILE="dataset_${foldname}.json"

mkdir -p ${RESULTS_DIRECTORY}

python /opt/3DUX-Net/main_train.py \
--root . \
--dataset_json_basename $JSON_FILE \
--out_classes 7 \
--output  $RESULTS_DIRECTORY \
--dataset stp \
--network 3DUXNET --mode train \
--batch_size 2 --crop_sample 1 --lr 0.0001 --optim AdamW --max_iter 171250 \
--eval_step 342 --cache_rate 0.01 --num_workers 4 \
--pretrain True --pretrained_weights $weight_file
