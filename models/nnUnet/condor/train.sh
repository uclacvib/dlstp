#!/bin/bash

echo fold $@

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export nnUNet_raw="/redactedsharedrive/pteng/pteng-public/dataset/stp/nnUNet_raw"
export nnUNet_preprocessed="/redactedsharedrive/pteng/pteng-public/dataset/stp/nnUNet_preprocessed"
export nnUNet_results="/redactedsharedrive/pteng/pteng-public/dataset/stp/nnUNet_results"
export nnUNet_n_proc_DA=8
export TORCH_HOME=/tmp/.torch
export XDG_CACHE_HOME=/tmp/.xdgcache
export TORCH_COMPILE_DEBUG_DIR=/tmp/.torchdebug
export TORCHINDUCTOR_CACHE_DIR=/tmp/.torchinductor
export TRITON_HOME=/tmp/.tritonhome
export TRITON_CACHE_DIR=/tmp/.triton

nnUNetv2_train 020 3d_fullres $@ --c

