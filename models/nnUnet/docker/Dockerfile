FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# https://github.com/MIC-DKFZ/nnUNet/issues/2680
COPY --chmod=666 polylr.py /opt/conda/lib/python3.11/site-packages/nnunetv2/training/lr_scheduler/polylr.py
COPY --chmod=666 predict_from_raw_data.py /opt/conda/lib/python3.11/site-packages/nnunetv2/inference/predict_from_raw_data.py
COPY --chmod=666 nnUNetTrainer.py /opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

WORKDIR /opt
COPY --chmod=666 inference.py /opt