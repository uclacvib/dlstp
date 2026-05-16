FROM projectmonai/monai:0.9.0

WORKDIR /opt

RUN git clone https://github.com/MASILab/3DUX-Net.git && \
    cd /opt/3DUX-Net && \
    git checkout 14ea46b7b4c4980b46aba066aaaa24b1d9c1bb0d

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY --chmod=666 load_datasets_transforms.py /opt/3DUX-Net
COPY --chmod=666 main_train.py /opt/3DUX-Net
COPY --chmod=666 inference.py /opt