"""

references

https://github.com/Project-MONAI/research-contributions/tree/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/UNETR/BTCV#testing
https://github.com/Project-MONAI/research-contributions/blob/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/UNETR/BTCV/test.py
https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/test.py
https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py
https://docs.monai.io/projects/monai-deploy-app-sdk/en/0.3.0/notebooks/tutorials/04_mis_tutorial.html

"""

import argparse
import os
import nibabel as nib

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
    MetaTensor,
)

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    Activationsd,
    AsDiscreted,
    Invertd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
)

def main(input_nifti_file_list,output_inference_folder,pretrained_pth,infer_overlap):
    
    out_dir = output_inference_folder
    basename = "pred"

    # `pre_transforms` copied from `train.py`
    pre_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Orientationd(keys=["img"], axcodes="RAS"),
            Spacingd(
                keys=["img"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(keys=["img"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=pre_transforms,
                orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(keys="pred", output_dir=out_dir, output_postfix=basename, resample=False),
        ]
    )
 
    batch_size = 1
    file_list = [{'img':x} for x in input_nifti_file_list]
    dataset = Dataset(data=file_list, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # `model = UNETR` copied from `train.py`
    model = UNETR(
        in_channels=1,
        out_channels=7,
        img_size=(96, 96, 96),
        feature_size=12,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    model_dict = torch.load(pretrained_pth,weights_only=True)
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        for d in dataloader:
            val_inputs = d["img"].cuda()
            d["pred"] = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=infer_overlap)
            d = [post_transforms(i) for i in decollate_batch(d)]

import os
import sys
import json
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_nifti_file')
    parser.add_argument('output_folder')
    parser.add_argument('pretrained_pth')
    parser.add_argument('--fold-str',type=str,default='fold_0',choices=['fold_0','fold_1','fold_2','fold_3','fold_4'])

    args = parser.parse_args()
    input_nifti_file = args.input_nifti_file
    output_folder = args.output_folder
    pretrained_pth = args.pretrained_pth
    fold_str = args.fold_str

    os.makedirs(output_folder,exist_ok=True)

    infer_overlap = 0.5
    input_nifti_file_list = [input_nifti_file]

    for input_file in input_nifti_file_list:
        basename = os.path.basename(input_file)
        casename = basename.replace(".nii.gz","")
        output_file = os.path.join(output_folder,casename,f"{casename}_pred.nii.gz")
        if os.path.exists(output_file):
            raise ValueError("file exists no need for inference!")

    main(input_nifti_file_list,output_folder,pretrained_pth,infer_overlap)

