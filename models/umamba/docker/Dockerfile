FROM nvcr.io/nvidia/pytorch:25.05-py3

RUN apt-get update \
    && apt-get install --no-install-recommends -y acl git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# The base image enforces versions for things like pytest for no good reason.
ENV PIP_CONSTRAINT=""
# There is no pre-build mamba image for pytorch 2.8, we build it before the rest to avoid rebuilds.
# We need to compile from the repo because of https://github.com/state-spaces/mamba/issues/720 (same for causal-conv1d)
# We set the number of workers to avoid OOM when compiling on laptop. (TODO: Can we make it configurable?)
RUN MAX_JOBS=2 pip install --no-build-isolation  "causal-conv1d@git+https://github.com/Dao-AILab/causal-conv1d@2a288a1"
RUN MAX_JOBS=2 pip install --no-build-isolation "mamba_ssm[causal-conv1d]@git+https://github.com/state-spaces/mamba@4a8a2a2"

WORKDIR /opt
RUN git clone https://github.com/bowang-lab/U-Mamba.git
RUN cd /opt/U-Mamba/umamba \
    && git checkout 28459e33ca03769800dd35e23c6e62491d1925b5 \
    && pip install -e .

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY --chmod=666 paths.py /opt/U-Mamba/umamba/nnunetv2/paths.py
COPY --chmod=666 polylr.py /opt/U-Mamba/umamba/nnunetv2/training/lr_scheduler/polylr.py
COPY --chmod=666 predict_from_raw_data.py /opt/U-Mamba/umamba/nnunetv2/inference/predict_from_raw_data.py
COPY --chmod=666 nnUNetTrainer.py /opt/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

WORKDIR /opt
COPY --chmod=666 inference.py /opt

# ## example 2 lines provided if need to run as non-root
# RUN useradd -u 1000 ubuntu
# USER ubuntu