"""

references

https://github.com/Project-MONAI/research-contributions/tree/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/UNETR/BTCV#testing
https://github.com/Project-MONAI/research-contributions/blob/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/UNETR/BTCV/test.py
https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/test.py
https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py
https://docs.monai.io/projects/monai-deploy-app-sdk/en/0.3.0/notebooks/tutorials/04_mis_tutorial.html

"""

import os
import sys
import shutil
import argparse
import tempfile
import SimpleITK as sitk
import numpy as np
import pandas as pd

import nibabel as nib
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

def main(input_nifti_file_list,output_inference_folder,unetr_results,fold_int=4,infer_overlap=0.5):

    pretrained_pth = os.path.join(unetr_results,f"fold_{fold_int}/best_metric_model.pth")

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


def main_one(input_nifti_file,output_nifti_file,csv_file,unetr_results,fold_int=4,infer_overlap=0.5)

    input_nifti_file_list = [input_nifti_file]
    basename = os.path.basename(input_nifti_file)
    casename = basename.replace(".nii.gz","")
    
    with tempfile.TemporaryDirectory() as tempdir:
        output_file = os.path.join(tempdir,casename,f"{casename}_pred.nii.gz")
        main(input_nifti_file_list,tempdir,unetr_results,fold_int=fold_int,infer_overlap=infer_overlap)
        if not os.path.exists(output_file):
            raise ValueError(f"inference failed! unable to find prediction file: {output_file}")
        shutil.copy(output_file,output_nifti_file)

    pred_obj = sitk.ReadImage(output_nifti_file)
    pred = sitk.GetArrayFromImage(pred_obj)
    wlung = np.logical_or(pred==1,pred==2)
    progression_ratio = np.sum(pred==1)/np.sum(wlung)
    df = pd.DataFrame([{"model_name":"unetr","stp_ratio":progression_ratio}])
    df.to_csv(csv_file,index=False)

raise NotImplementedError()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_nifti_file')
    parser.add_argument('output_nifti_file')
    parser.add_argument('csv_file')
    parser.add_argument('unetr_results')
    parser.add_argument('--fold-int',type=int,default=4,choices=[0,1,2,3,4])

    args = parser.parse_args()
    input_nifti_file = args.input_nifti_file
    output_nifti_file = args.output_nifti_file
    csv_file = args.csv_file
    unetr_results = args.unetr_results
    fold_int = args.fold_int

    main_one(input_nifti_file,output_nifti_file,csv_file,unetr_results,fold_int)


"""
unetr_results = "/placeholder/dataset/stp/unetr_results"
pretrained_pth = f"/placeholder/dataset/stp/unetr_results/{fold_str}/best_metric_model.pth"

"""