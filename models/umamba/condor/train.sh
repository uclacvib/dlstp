#!/bin/bash

echo fold $@

export nnUNet_raw="/redactedsharedrive/pteng/pteng-public/dataset/stp/nnUNet_raw"
export nnUNet_preprocessed="/redactedsharedrive/pteng/pteng-public/dataset/stp/umamba_preprocessed"
export nnUNet_results="/redactedsharedrive/pteng/pteng-public/dataset/stp/umamba_results"

export TORCH_HOME=/tmp/.torch
export XDG_CACHE_HOME=/tmp/.xdgcache
export TORCH_COMPILE_DEBUG_DIR=/tmp/.torchdebug
export TORCHINDUCTOR_CACHE_DIR=/tmp/.torchinductor
export TRITON_HOME=/tmp/.tritonhome
export TRITON_CACHE_DIR=/tmp/.triton
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export nnUNet_n_proc_DA=4
export UMAMBA_FOLDER="/redactedsharedrive/pteng/pteng-public/dataset/stp/"
mkdir -p "/redactedsharedrive/pteng/pteng-public/dataset/stp/umamba_results"

nnUNetv2_train 020 3d_fullres $@ -tr nnUNetTrainerUMambaEncNoAMP --c

