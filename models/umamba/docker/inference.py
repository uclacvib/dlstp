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
os.environ['TORCH_HOME']='/tmp/.torch'
os.environ['XDG_CACHE_HOME']='/tmp/.xdgcache'
os.environ['TORCH_COMPILE_DEBUG_DIR']='/tmp/.torchdebug'
os.environ['TORCHINDUCTOR_CACHE_DIR']='/tmp/.torchinductor'
os.environ['TRITON_HOME']='/tmp/.tritonhome'
os.environ['TRITON_CACHE_DIR']='/tmp/.triton'
os.environ['UMAMBA_FOLDER']='None'
os.environ['nnUNet_results']='None'
os.environ['nnUNet_raw']='None'

import os
import sys
import shutil
import argparse
import tempfile
import SimpleITK as sitk
import numpy as np
import pandas as pd

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main(nnUNet_results,input_list_of_list,output_list,fold_int):
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
                                 output_list,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

def main_one(input_nifti_file,output_nifti_file,csv_file,umamba_results,fold_int=0):
    input_list_of_list = [[input_nifti_file]]
    output_list = [output_nifti_file]
    main(umamba_results,input_list_of_list,output_list,fold_int)

    pred_obj = sitk.ReadImage(output_nifti_file)
    pred = sitk.GetArrayFromImage(pred_obj)
    wlung = np.logical_or(pred==1,pred==2)
    progression_ratio = np.sum(pred==1)/np.sum(wlung)
    df = pd.DataFrame([{"model_name":"unetr","stp_ratio":progression_ratio}])
    df.to_csv(csv_file,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_nifti_file')
    parser.add_argument('output_nifti_file')
    parser.add_argument('csv_file')
    parser.add_argument('umamba_results')
    parser.add_argument('--fold-int',type=int,default=0,choices=[0,1,2,3,4])

    args = parser.parse_args()
    input_nifti_file = args.input_nifti_file
    output_nifti_file = args.output_nifti_file
    csv_file = args.csv_file
    umamba_results = args.umamba_results
    fold_int = args.fold_int

    main_one(input_nifti_file,output_nifti_file,csv_file,umamba_results,fold_int=fold_int)

"""
umamba_results = '/placeholder/stp/umamba_results/Dataset020_STP/nnUNetTrainerUMambaEncNoAMP__nnUNetPlans__3d_fullres'
"""