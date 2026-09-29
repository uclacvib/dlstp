
import os
from huggingface_hub import snapshot_download

HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
model_repo_folder = snapshot_download(repo_id="cvib/dlstp-unetr-model",repo_type='model',token=HF_ACCESS_TOKEN)