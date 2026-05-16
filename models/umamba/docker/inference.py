"""
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
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
os.environ['nnUNet_n_proc_DA']="4"
os.environ['TRITON_HOME']='None'
os.environ['TRITON_CACHE_DIR']='None'
os.environ['UMAMBA_FOLDER']='None'
os.environ['nnUNet_results']='None'
os.environ['nnUNet_raw']='None'

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main(nnUNet_results,input_list_of_list,output_list_of_list,fold_int):
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        nnUNet_results,
        use_folds=(fold_int,), # !!!
        checkpoint_name='checkpoint_best.pth', # ? _final.pth
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(input_list_of_list,
                                 output_list_of_list,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

import os
import sys
import json
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_nifti_file')
    parser.add_argument('output_folder')
    parser.add_argument('umamba_results')
    parser.add_argument('--fold-int',type=int,default=0,choices=[0,1,2,3,4])

    args = parser.parse_args()
    input_nifti_file = args.input_nifti_file
    output_folder = args.output_folder
    umamba_results = args.umamba_results
    fold_int = args.fold_int

    input_list_of_list = [[input_nifti_file]]
    os.makedirs(output_folder,exist_ok=True)
    output_list = [
        os.path.join(output_folder,os.path.basename(x[0])) for x in input_list_of_list
    ]
    for x in output_list:
        if os.path.exists(x):
            raise ValueError("file exists no need for inference!")

    main(umamba_results,input_list_of_list,output_list,fold_int)
