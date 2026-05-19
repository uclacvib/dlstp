
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

def main(nnUNet_results,input_list_of_list,output_list,fold_int=0):
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
    predictor.predict_from_files(input_list_of_list, # join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
                                 output_list, # join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

def main_one(input_nifti_file,output_nifti_file,csv_file,nnUNet_results,fold_int=0):
    input_list_of_list = [[input_nifti_file]]
    output_list = [output_nifti_file]
    main(nnUNet_results,input_list_of_list,output_list,fold_int)

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
    parser.add_argument('nnUNet_results')
    parser.add_argument('--fold-int',type=int,default=0,choices=[0,1,2,3,4])

    args = parser.parse_args()
    input_nifti_file = args.input_nifti_file
    output_nifti_file = args.output_nifti_file
    csv_file = args.csv_file
    nnUNet_results = args.nnUNet_results
    fold_int = args.fold_int

    main_one(input_nifti_file,output_nifti_file,csv_file,nnUNet_results,fold_int=fold_int)


"""

nnUNet_results = '/placeholder/dataset/stp/nnUNet_results/Dataset020_STP/nnUNetTrainer__nnUNetPlans__3d_fullres'

"""